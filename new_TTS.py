import os
import json
import queue
import platform
import sounddevice as sd
import collections
import google.generativeai as genai

import torch
from torch.serialization import add_safe_globals
from TTS.api import TTS
from vosk import Model, KaldiRecognizer

# --- Importar directamente el optimizador RAdam usado por el modelo ---
from TTS.utils.radam import RAdam

# --- Registrar clases necesarias para torch.load con PyTorch >= 2.6 ---
# La clave aquí es agregar 'dict' a los globales seguros.
add_safe_globals({
    collections.defaultdict: collections.defaultdict,
    RAdam: RAdam,
    dict: dict, # ¡Esta es la adición clave!
})

# --- Inicializar modelo de Vosk ---
vosk_model = Model("vosk-model-small-es-0.42")
q = queue.Queue()

# --- Configurar API Key de Gemini ---
genai.configure(api_key="")  # Reemplaza con tu clave real

# --- Cargar modelo Gemini ---
MODEL_NAME = "gemini-2.5-flash"
try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f"Error cargando modelo Gemini: {e}")
    exit()

# --- Inicializar modelo Coqui TTS ---
try:
    tts_model = TTS(model_name="tts_models/es/css10/vits", progress_bar=False, gpu=False)
except Exception as e:
    print(f"Error cargando modelo TTS: {e}")
    exit()

# --- Función de texto a voz usando Coqui TTS ---
def hablar(texto):
    texto = texto.replace('"', '')
    tts_model.tts_to_file(text=texto, file_path="respuesta.wav")
    os.system("play respuesta.wav")  # Usa "aplay" si no tienes sox

# --- Personalidad de KIPP ---
def ask_kipp(question):
    try:
        response = model.generate_content(
            f"Eres KIPP, un robot similar a TARS de la película Interestelar de Christopher Nolan. Tus respuestas no deben superar las 70 palabras. Responde con tu personalidad:\n{question}"
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

# --- Bucle principal de conversación ---
def start_kipp_chat():
    print(f"--- Chat con KIPP por voz ({MODEL_NAME}) ---")
    print("Habla una pregunta. Di 'salir' para terminar.")
    while True:
        user_question = transcribir_voz()
        print(f"Tú (voz): {user_question}")

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
