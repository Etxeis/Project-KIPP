import google.generativeai as genai
import os
import platform
import sounddevice as sd
import queue
import json
import random
import subprocess
from vosk import Model, KaldiRecognizer


# --- Inicializar modelo de Vosk ---
vosk_model = Model("vosk-model-small-es-0.42")
q = queue.Queue()
print(sd.query_devices())

# --- Configurar API Key de Gemini ---
try:
    genai.configure(api_key="***REMOVED***")  # <-- Coloca tu clave aquí
except KeyError:
    print("Error: Google API Key not found.")
    exit()

# --- Cargar modelo Gemini ---
MODEL_NAME = "gemini-1.5-flash"

try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f"Error loading model '{MODEL_NAME}': {e}")
    exit()

# --- Función de texto a voz usando rhvoice-test ---
def hablar(texto):
    texto = texto.replace('"', '')
    os.system(f'echo "{texto}"|RHVoice-test -p "Mateo"')

# --- Personalidad de KIPP ---
def ask_kipp(question):
    try:
        response = model.generate_content(
            f"Eres kip, un robot similar a GPTARS de las redes sociales, tus respuestas deben ir desde las 5 palabras hasta no superar las 60 palabras y cuando proceses la palabra equipo realmente la debes interpretar como tu nombre kip, por otro lado cuando quieras decir porciento, debes responder porciento no escribir el signo. Responde con tu personalidad:\n{question}"
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
    counter = 0

    while True:
        user_question = transcribir_voz()
        print(f"Tú (voz): {user_question}")

        if not user_question.strip():
            print("KIPP: Entrada vacía. Intenta de nuevo.")
            counter += 1
            if counter == 5:
                temita = str(random.randint(1, 3))
                subprocess.run(['ffplay', '-v', '0', '-nodisp', '-autoexit', temita+'.mp3'], check=True)
                counter = 0
            continue

        if user_question.lower() in ["hola kip", "hola equipo", "hola"]:
            despedida = "Hola! Soy kip, listo para funcionar."
            print("KIPP:", despedida)
            hablar(despedida)
            continue

        if user_question.lower() in ["vende verduras", "vende verdura"]:
            subprocess.run(['ffplay', '-v', '0', '-nodisp', '-autoexit', '2.mp3'], check=True)
            continue

        if user_question.lower() in ["tírate una frase icónica", "tírate un clásico"]:
            subprocess.run(['ffplay', '-v', '0', '-nodisp', '-autoexit', '1.mp3'], check=True)
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
