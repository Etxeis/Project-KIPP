import google.generativeai as genai
import os
import platform
import sounddevice as sd
import queue
import json
import subprocess
import time
import numpy as np # <-- ¡NUEVA IMPORTACIÓN!

# Importar las bibliotecas de Picovoice
from pvorca import Orca, OrcaActivationLimitError, OrcaInvalidArgumentError
# import struct # Ya no necesitamos struct

# --- Configuración de Picovoice Orca ---

# Ruta de la librería nativa para Raspberry Pi 4B (Cortex-A72).
# Se asume OS de 64 bits (aarch64), que es lo más común para Pi 4B.
# SI TU OS ES DE 32 BITS, DEBERÍAS CAMBIAR ESTA RUTA A:
# "/home/pablo/Project-KIPP/Project-KIPP/lib/python3.11/site-packages/pvorca/lib/raspberry-pi/cortex-a72/libpv_orca.so"
# Puedes verificar tu arquitectura con: dpkg --print-architecture
PICOVOICE_LIBRARY_PATH = "/home/pablo/Project-KIPP/Project-KIPP/lib/python3.11/site-packages/pvorca/lib/raspberry-pi/cortex-a72-aarch64/libpv_orca.so"

# Acceso a la clave de Picovoice (siempre mejor como variable de entorno)
PICOVOICE_ACCESS_KEY = os.environ.get("PICOVOICE_ACCESS_KEY")
if not PICOVOICE_ACCESS_KEY or PICOVOICE_ACCESS_KEY == "YOUR_PICOVOICE_ACCESS_KEY_HERE":
    # ADVERTENCIA: NO INCORPORAR EN PRODUCCIÓN ESTO!
    # ¡REEMPLAZA "YOUR_PICOVOICE_ACCESS_KEY_HERE" con tu AccessKey REAL de Picovoice Console!
    PICOVOICE_ACCESS_KEY = ""
    print("Advertencia: Picovoice AccessKey obtenida directamente del código. Considera usar variables de entorno.")


# Ruta al modelo de voz de Orca (español masculino: mateo)
# Asegúrate de que el archivo 'orca_params_es_male.pv' esté en la carpeta 'orca_models'
orca_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orca_models", "orca_params_es_male.pv")


# --- Inicializar modelo de Vosk ---
from vosk import Model, KaldiRecognizer
vosk_model = Model("vosk-model-small-es-0.42")
q = queue.Queue()

# --- Configurar API Key de Gemini ---
try:
    api_key_gemini = os.environ.get("GEMINI_API_KEY")
    if not api_key_gemini:
        # Si no está en variable de entorno, úsala directamente (SOLO PARA PRUEBAS)
        # ¡ADVERTENCIA: NO INCORPORAR EN PRODUCCIÓN ESTO!
        api_key_gemini = "" # <-- ¡REEMPLAZA CON TU API Key REAL de Gemini!
        print("Advertencia: API Key de Gemini obtenida directamente del código. Considera usar variables de entorno.")
    genai.configure(api_key=api_key_gemini)
except KeyError:
    print("Error: Google API Key not found. Asegúrate de que esté configurada.")
    exit()

# --- Cargar modelo Gemini ---
MODEL_NAME = "gemini-2.5-flash"

try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f"Error cargando el modelo '{MODEL_NAME}': {e}")
    exit()

# --- Inicializar Orca (una vez al inicio del programa) ---
# Es más eficiente inicializar Orca una sola vez y reutilizar la instancia.
orca_engine = None
try:
    orca_engine = Orca(
        access_key=PICOVOICE_ACCESS_KEY,
        model_path=orca_model_path,
        library_path=PICOVOICE_LIBRARY_PATH # ¡Argumento crítico!
    )
    print(f"Picovoice Orca engine initialized with sample rate: {orca_engine.sample_rate}")
except OrcaInvalidArgumentError as e:
    print(f"Error de argumento al inicializar Orca: {e}")
    print("Asegúrate de que el AccessKey es válido, el model_path es correcto, y el library_path apunta al archivo .so correcto.")
    exit()
except OrcaActivationLimitError as e:
    print(f"Error de límite de activación de Orca: {e}")
    print("Tu AccessKey ha excedido su límite de uso gratuito. Inicia sesión en Picovoice Console para ver los detalles.")
    exit()
except Exception as e:
    print(f"Error inesperado al inicializar Picovoice Orca: {e}")
    exit()


# --- Función de texto a voz usando Picovoice Orca (Modo Streaming) ---
# Esta función es un helper ahora, principalmente usada para la despedida.
# El streaming principal ocurre dentro de ask_kipp.
def hablar_con_orca_streaming(texto_completo):
    """
    Sintetiza texto a voz usando Picovoice Orca en modo streaming
    y lo reproduce directamente.
    """
    if orca_engine is None:
        print("Error: El motor Orca no está inicializado.")
        return

    buffer_size = 512 # Puedes ajustar este valor
    
    orca_stream = orca_engine.stream_open()

    print("KIPP (Orca): Reproduciendo...")
    try:
        with sd.OutputStream(samplerate=orca_engine.sample_rate, channels=1, dtype='int16', blocksize=buffer_size) as stream:
            words = texto_completo.split()
            current_chunk = ""
            for i, word in enumerate(words):
                current_chunk += word + " "
                if len(current_chunk) > 10 or i == len(words) - 1:
                    pcm = orca_stream.synthesize(current_chunk.strip())
                    if pcm is not None and len(pcm) > 0:
                        # Convertir a numpy array de int16
                        stream.write(np.array(pcm, dtype=np.int16)) # <-- ¡CAMBIO AQUÍ!
                    current_chunk = ""
            
            pcm = orca_stream.flush()
            if pcm is not None and len(pcm) > 0:
                 # Convertir a numpy array de int16
                 stream.write(np.array(pcm, dtype=np.int16)) # <-- ¡CAMBIO AQUÍ!

            stream.stop()
            stream.close()

    except Exception as e:
        print(f"Error al intentar que KIPP hable con Picovoice Orca: {e}")
    finally:
        orca_stream.close()


# --- Personalidad de KIPP ---
def ask_kipp(question):
    try:
        # Habilita el streaming de la respuesta de Gemini
        response_generator = model.generate_content(
            f"Eres KIPP, un robot similar a TARS de la película Interestelar de Christopher Nolan, tus respuestas no deben superar las 70 palabras. Responde con tu personalidad:\n{question}",
            stream=True
        )
        
        full_response = ""
        # Abrir un stream de Orca para esta respuesta específica
        orca_stream = orca_engine.stream_open()
        
        print("KIPP (Orca Stream): ", end="", flush=True) # Para ver el texto mientras se genera
        
        # Iniciar sounddevice stream para reproducir el audio de Orca
        with sd.OutputStream(samplerate=orca_engine.sample_rate, channels=1, dtype='int16', blocksize=512) as audio_stream:
            for chunk in response_generator:
                text_chunk = chunk.text
                full_response += text_chunk
                print(text_chunk, end="", flush=True) # Imprime el texto de Gemini

                # Envía el chunk de texto de Gemini a Orca para síntesis
                pcm = orca_stream.synthesize(text_chunk)
                if pcm is not None and len(pcm) > 0:
                    # Convertir a numpy array de int16
                    audio_stream.write(np.array(pcm, dtype=np.int16)) # <-- ¡CAMBIO AQUÍ!
            
            # Asegurar que cualquier audio restante en el buffer de Orca se reproduzca
            pcm = orca_stream.flush()
            if pcm is not None and len(pcm) > 0:
                # Convertir a numpy array de int16
                audio_stream.write(np.array(pcm, dtype=np.int16)) # <-- ¡CAMBIO AQUÍ!

        # Cierra el stream de Orca para este turno de conversación
        orca_stream.close()
        print() # Nueva línea después de la respuesta completa
        return full_response.strip()

    except Exception as e:
        print(f"Error en ask_kipp con Picovoice Orca: {e}")
        return f"KIPP: Error de comunicación con la IA. Detalles: {e}"


# --- Captura de voz con Vosk ---
def callback(indata, frames, time, status):
    if status:
        print("⚠️", status)
    q.put(bytes(indata))


def transcribir_voz():
    print("🎙️ Habla ahora...")
    try:
        with sd.RawInputStream(samplerate=48000, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            recognizer = KaldiRecognizer(vosk_model, 48000)
            texto_final = ""
            while True:
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    texto_final = result.get("text", "")
                    break
            return texto_final
    except Exception as e:
        print(f"Error al inicializar o usar el micrófono: {e}")
        return ""


# --- Bucle principal de conversación ---
def start_kipp_chat():
    print(f"--- Chat con KIPP por voz ({MODEL_NAME} + Orca) ---")
    print("Habla una pregunta. Di 'salir' para terminar.")

    while True:
        user_question = transcribir_voz()
        print(f"Tú (voz): {user_question}")

        if not user_question.strip():
            print("KIPP: Entrada vacía. Intenta de nuevo.")
            continue

        if user_question.lower() in ["salir", "exit", "quit", "terminar sesión"]:
            despedida = "Terminando sesión. Adiós humano."
            print("KIPP:", despedida)
            ask_kipp(despedida)
            break

        kipp_answer = ask_kipp(user_question)


# --- Ejecutar ---
if __name__ == "__main__":
    try:
        # Asegúrate de que numpy esté instalado: pip install numpy
        start_kipp_chat()
    finally:
        if orca_engine is not None:
            orca_engine.delete()
            print("Picovoice Orca engine resources released.")
