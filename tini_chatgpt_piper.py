import google.generativeai as genai
import os
import platform
import sounddevice as sd
import queue
import json

# --- Importaciones para Piper TTS (librería Python) ---
from piper import PiperVoice, SynthesisConfig # Importa PiperVoice y SynthesisConfig
import soundfile as sf       # Necesario para manejar el audio de Piper y sounddevice
from io import BytesIO       # Para trabajar con datos de audio en memoria

from vosk import Model, KaldiRecognizer


# --- Inicializar modelo de Vosk ---
vosk_model = Model("vosk-model-small-es-0.42")
q = queue.Queue()

# --- Configurar API Key de Gemini ---
try:
    # --- COLOCA TU CLAVE API DE GEMINI AQUÍ ---
    genai.configure(api_key="")
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

# --- Cargar la voz de Piper una única vez al inicio del script ---
# Esto es crucial para la eficiencia y para que el modelo esté listo.
try:
    # Usamos el identificador 'es_MX-ald-medium' ya que confirmamos que está disponible.
    piper_voice = PiperVoice.load_voice_from_pretrained("es_MX-ald-medium")
    print("Voz de Piper (Ald) cargada con éxito usando la librería.")
except Exception as e:
    print(f"Error al cargar la voz de Piper (Ald): {e}")
    print("POSIBLES SOLUCIONES:")
    print("1. Asegúrate de que 'piper-tts' esté actualizado: `pip install --upgrade piper-tts`")
    print("2. Descarga los modelos de voz: `python -m piper.download_voices` (dentro de tu entorno virtual)")
    print("3. Asegúrate de tener conexión a internet la primera vez que se carga un modelo.")
    exit()

# --- Configuración de síntesis para Piper ---
# Puedes ajustar estos valores para modificar la voz de KIPP.
# Experimenta con ellos para encontrar el sonido que más te guste.
# volumen: 0.0 (silencio) a 1.0 (máximo). 0.5 es la mitad.
# length_scale: 1.0 es la velocidad normal. >1.0 más lento, <1.0 más rápido. (Ej: 1.2 es un poco más lento)
# noise_scale: Cuánta variación en el tono y la entonación. 0.6 es un buen punto de partida.
# noise_w_scale: Variación en la cadencia del habla. 0.6 es un buen punto de partida.
# normalize_audio: Normalmente True para asegurar volumen consistente. False si quieres el audio 'crudo'.
KIPP_SYNTHESIS_CONFIG = SynthesisConfig(
    volume=0.8,         # Un poco más bajo que el máximo por defecto
    length_scale=1.1,   # Ligeramente más lento para mayor claridad
    noise_scale=0.6,    # Un poco de variación para que no suene tan monótono
    noise_w_scale=0.6,  # Un poco de variación en la velocidad de las palabras
    normalize_audio=True # Mantiene el volumen consistente
)


# --- Función de texto a voz usando la librería Piper TTS ---
def hablar(texto):
    """
    Sintetiza texto a voz usando la librería Python 'piper-tts'
    con la configuración de síntesis definida y lo reproduce directamente con sounddevice.
    """
    try:
        # Generar los bytes de audio en formato raw (PCM de 16bit, 16kHz)
        # Pasamos la configuración de síntesis aquí.
        audio_bytes_io = BytesIO()
        for chunk in piper_voice.synthesize_stream_raw(texto, syn_config=KIPP_SYNTHESIS_CONFIG):
            audio_bytes_io.write(chunk)
        
        audio_bytes_io.seek(0) # Volver al inicio del buffer en memoria
        
        # Leer los datos de audio con soundfile desde el buffer
        data, samplerate = sf.read(audio_bytes_io, format='RAW', samplerate=16000, channels=1, subtype='PCM_16')
        
        # Reproducir el audio con sounddevice
        sd.play(data, samplerate)
        sd.wait() # Esperar a que termine la reproducción
        
    except Exception as e:
        print(f"Error al intentar que KIPP hable con Piper: {e}")
        print("Verifica si el modelo de voz fue cargado correctamente al inicio del script.")


# --- Personalidad de KIPP ---
def ask_kipp(question):
    try:
        response = model.generate_content(
            f"Eres KIPP, un robot similar a TARS de la película Interestelar de Christopher Nolan, tus respuestas no deben superar las 70 palabras. Responde con tu personalidad:\n{question}"
        )
        return response.text.strip()
    except Exception as e:
        return f"KIPP: Error de comunicación con la IA. Detalles: {e}"

# --- Captura de voz con Vosk (LÓGICA ORIGINAL MANTENIDA) ---
def callback(indata, frames, time, status):
    if status:
        print("⚠️", status)
    q.put(bytes(indata))


def transcribir_voz():
    print("🎙️ Habla ahora...")
    try:
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            recognizer = KaldiRecognizer(vosk_model, 16000)
            texto_final = ""
            while True:
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    texto_final = result.get("text", "")
                    break
            return texto_final
    except Exception as e:
        print(f"Error al inicializar o usar el micrófono (Vosk): {e}")
        return "" # Retorna una cadena vacía para evitar que el programa se detenga si el micrófono falla


# --- Bucle principal de conversación ---
def start_kipp_chat():
    print(f"--- Chat con KIPP por voz ({MODEL_NAME}) ---")
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
            hablar(despedida)
            break

        kipp_answer = ask_kipp(user_question)
        # Limpiar la respuesta de Gemini para la voz (quitar asteriscos, comillas, etc.)
        kipp_answer_clean = kipp_answer.replace('*', '').replace('"', '').strip()

        print("KIPP:", kipp_answer)
        hablar(kipp_answer_clean)


# --- Ejecutar ---
if __name__ == "__main__":
    start_kipp_chat()
