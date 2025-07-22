import google.generativeai as genai
import os
import platform
import sounddevice as sd
import queue
import json
import subprocess

from vosk import Model, KaldiRecognizer

# --- Inicializar modelo de Vosk ---
vosk_model = Model("vosk-model-small-es-0.42")
q = queue.Queue()

# --- Configurar API Key de Gemini ---
try:
    api_key_gemini = os.environ.get("GEMINI_API_KEY")
    if not api_key_gemini:
        api_key_gemini = "" # ¡ADVERTENCIA: NO INCORPORAR EN PRODUCCIÓN!
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

# --- Función de texto a voz usando Piper ---
def hablar(texto):
    """
    Sintetiza texto a voz usando el comando 'piper' del sistema,
    ubicado dentro de una subcarpeta llamada 'piper'.
    """
    texto = texto.replace('"', '')

    # Definir la ruta al ejecutable piper y al modelo
    # Asumiendo que 'piper' el ejecutable está en la subcarpeta 'piper_folder'
    # y el modelo 'es_ES-davefx-medium.onnx' también está en esa subcarpeta.
    
    # Construye la ruta absoluta al directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define el nombre de la subcarpeta que contiene el ejecutable y el modelo
    piper_folder_name = "piper" # <-- ASUME QUE LA CARPETA SE LLAMA 'piper'
    
    # Construye la ruta completa al ejecutable piper
    piper_executable_path = os.path.join(script_dir, piper_folder_name, "piper")
    
    # Construye la ruta completa al archivo del modelo .onnx
    model_path = os.path.join(script_dir, piper_folder_name, "es_ES-davefx-medium.onnx")

    try:
        # Llama a 'piper' usando la ruta completa al ejecutable y al modelo
        subprocess.run(
            [piper_executable_path, '--model', model_path, '--output_file', 'output.wav'],
            input=texto.encode('utf-8'),
            check=True,
            capture_output=True # Captura stdout/stderr para depuración
        )

        # Reproducir el archivo WAV generado
        subprocess.run(['aplay', 'output.wav'], check=True)

    except FileNotFoundError as e:
        print(f"Error: Comando no encontrado. Asegúrate de que '{piper_executable_path}' (y su modelo) o 'aplay' estén en su lugar y con los permisos correctos.")
        print(f"Detalles: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error al intentar que KIPP hable con Piper. Código de salida: {e.returncode}")
        print(f"Salida estándar (stdout): {e.stdout.decode('utf-8') if e.stdout else 'N/A'}")
        print(f"Salida de error (stderr): {e.stderr.decode('utf-8') if e.stderr else 'N/A'}")
        print("Asegúrate de que el archivo .onnx del modelo de Piper sea correcto y esté en la ruta esperada.")
    except Exception as e:
        print(f"Un error inesperado ocurrió al intentar hablar con Piper: {e}")


# --- Personalidad de KIPP ---
def ask_kipp(question):
    try:
        response = model.generate_content(
            f"Eres kip, un robot similar a TARS de la película Interestelar de Christopher Nolan, tus respuestas deben ir desde las 5 palabras hasta no superar las 70 palabras y cuando proceses la palabra equipo realmente la debes interpretar como tu nombre kip, por otro lado cuando quieras decir porciento, debes responder porciento no escribir el signo. Responde con tu personalidad:\n{question}"
        )
        return response.text.strip()
    except Exception as e:
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
    print(f"--- Chat con KIPP por voz ({MODEL_NAME}) ---")
    print("Habla una pregunta. Di 'salir' para terminar.")

    while True:
        user_question = transcribir_voz()
        print(f"Tú (voz): {user_question}")

        if not user_question.strip():
            print("KIPP: Entrada vacía. Intenta de nuevo.")
            continue

        if user_question.lower() in ["oye equipo", "kip", "equipo", "oye kip", "oye"]:
            despedida = "Dime."
            print("KIPP:", despedida)
            hablar(despedida)
            continue

        if user_question.lower() in ["salir", "exit", "quit", "terminar sesión"]:
            despedida = "Terminando sesión. Adiós humano."
            print("KIPP:", despedida)
            hablar(despedida)
            break

        kipp_answer = ask_kipp(user_question)
        print("KIPP:", kipp_answer)
        hablar(kipp_answer)


# --- Ejecutar ---
if __name__ == "__main__":
    start_kipp_chat()
