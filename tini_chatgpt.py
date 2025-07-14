import google.generativeai as genai
import os
import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer

# --- Inicializar modelo de Vosk (español) ---
vosk_model = Model("vosk-model-small-es-0.42")
q = queue.Queue()

# --- Configurar clave de API de Gemini ---
try:
    genai.configure(api_key="")  # ← Reemplaza con tu API key real
except KeyError:
    print("Error: API Key de Gemini no encontrada.")
    exit()

# --- Cargar modelo de Gemini ---
MODEL_NAME = "gemini-2.5-flash"
try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f"Error al cargar el modelo '{MODEL_NAME}': {e}")
    exit()

# --- Función de texto a voz con espeak ---
def hablar(texto):
    texto = texto.replace('"', '')  # Limpia comillas que pueden romper el comando
    os.system(f'espeak -s 130 -v es-la "{texto}" --stdout | aplay')

# --- Generar respuesta con personalidad KIPP ---
def ask_kipp(question):
    try:
        response = model.generate_content(
            f"Eres KIPP, el robot sarcástico y sincero de Interstellar. Responde con tu personalidad:\n{question}"
        )
        return response.text.strip()
    except Exception as e:
        return f"KIPP: Error de comunicación con la IA. Detalles: {e}"

# --- Callback para entrada de audio ---
def callback(indata, frames, time, status):
    if status:
        print("⚠️", status)
    q.put(bytes(indata))

# --- Transcribir voz a texto ---
def transcribir_voz():
    print("🎙️ Habla ahora...")
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

# --- Bucle de conversación con KIPP ---
def start_kipp_chat():
    print(f"--- Chat con KIPP por voz ({MODEL_NAME}) ---")
    print("Habla una pregunta. Di 'salir' para terminar.")

    while True:
        user_question = transcribir_voz()
        print(f"Tú (voz): {user_question}")

        if not user_question.strip():
            print("KIPP: Entrada vacía. Intenta de nuevo.")
            continue

        if user_question.lower() in ["salir", "exit", "quit"]:
            despedida = "Terminando sesión. Adiós humano."
            print("KIPP:", despedida)
            hablar(despedida)
            break

        kipp_answer = ask_kipp(user_question)
        print("KIPP:", kipp_answer)
        hablar(kipp_answer)

# --- Ejecutar programa ---
if __name__ == "__main__":
    start_kipp_chat()
