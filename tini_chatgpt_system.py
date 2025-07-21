import google.generativeai as genai
import os
import platform
import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer


# --- Inicializar modelo de Vosk ---
vosk_model = Model("vosk-model-small-es-0.42")
q = queue.Queue()
print(sd.query_devices())

# --- Configurar API Key de Gemini ---
try:
    genai.configure(api_key="")  # <-- Coloca tu clave aquí
except KeyError:
    print("Error: Google API Key not found.")
    exit()

# --- Cargar modelo Gemini ---
MODEL_NAME = "gemini-2.5-flash"

try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f"Error loading model '{MODEL_NAME}': {e}")
    exit()

# --- Función de texto a voz usando rhvoice-test ---
def hablar(texto):
    texto = texto.replace('"', '')
    os.system(f'echo "{texto}"|rhvoice.test -p "Mateo"')

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
        print("KIPP:", kipp_answer)
        hablar(kipp_answer)


# --- Ejecutar ---
if __name__ == "__main__":
    start_kipp_chat() 
