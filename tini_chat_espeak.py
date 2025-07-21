import google.generativeai as genai
import os
import platform
import sounddevice as sd
import queue
import json
import subprocess # Necesario para ejecutar comandos de sistema como 'espeak-ng'
from vosk import Model, KaldiRecognizer


# --- Inicializar modelo de Vosk ---
# Asegúrate de que el modelo de Vosk esté en la ruta correcta.
# Por ejemplo, si está en el mismo directorio que el script o una subcarpeta relativa.
vosk_model = Model("vosk-model-small-es-0.42")
q = queue.Queue()

# --- Configurar API Key de Gemini ---
try:
    # --- COLOCA TU CLAVE API DE GEMINI AQUÍ ---
    # Es ALTAMENTE recomendable no incrustar la API key directamente en el código.
    # Considera usar una variable de entorno: os.environ.get("GEMINI_API_KEY")
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

# --- Función de texto a voz usando eSpeak-NG (REEMPLAZO DE RHVOICE) ---
def hablar(texto):
    """
    Sintetiza texto a voz usando el comando 'espeak-ng' del sistema.
    Ajusta para usar la voz en español (-v es) y una velocidad (-s 150).
    """
    # Elimina comillas del texto para evitar problemas con el comando de shell
    # Aunque subprocess.run con lista de argumentos es más seguro, sigue siendo buena práctica.
    texto = texto.replace('"', '') 
    
    try:
        # Llama a 'espeak-ng' con los argumentos necesarios
        # -v es: usa la voz de español.
        # -s 150: establece la velocidad a 150 palabras por minuto.
        # check=True: hará que Python genere un error si el comando 'espeak-ng' falla.
        subprocess.run(['espeak-ng', '-v', 'es', '-s', '150', texto], check=True)
        # print(f"KIPP (eSpeak): {texto}") # Para depuración: descomentar para ver qué habla
    except FileNotFoundError:
        print("Error: El comando 'espeak-ng' no se encontró. Asegúrate de que eSpeak-NG esté instalado y en tu PATH.")
        print("Puedes instalarlo con: sudo apt install -y espeak-ng")
    except subprocess.CalledProcessError as e:
        print(f"Error al intentar que KIPP hable con eSpeak-NG: {e}")
        print(f"Salida de error de eSpeak-NG: {e.stderr.decode('utf-8')}")
    except Exception as e:
        print(f"Un error inesperado ocurrió al intentar hablar con eSpeak-NG: {e}")


# --- Personalidad de KIPP ---
def ask_kipp(question):
    try:
        response = model.generate_content(
            f"Eres KIPP, un robot similar a TARS de la película Interestelar de Christopher Nolan, tus respuestas no deben superar las 70 palabras. Responde con tu personalidad:\n{question}"
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
    # Asegúrate de que samplerate y blocksize coincidan con la capacidad de tu micrófono y el modelo Vosk.
    # Si sigues teniendo problemas con el micrófono, puede que necesites especificar 'device'.
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
        # Si hay un error con el micrófono, podrías querer manejarlo mejor,
        # por ejemplo, pidiendo al usuario que verifique la conexión.
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
            hablar(despedida) # Usar la función 'hablar' (con eSpeak-NG)
            break

        kipp_answer = ask_kipp(user_question)
        print("KIPP:", kipp_answer)
        hablar(kipp_answer) # Usar la función 'hablar' (con eSpeak-NG)


# --- Ejecutar ---
if __name__ == "__main__":
    start_kipp_chat()
