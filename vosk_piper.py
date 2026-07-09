import json
import os
import queue
import random
import subprocess
import shutil
import threading
import time
import array
import numpy as np
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import google.generativeai as genai

# ================== CONFIG VOSK ==================
# Ruta al modelo de Vosk en español (carpeta descargada, no un .zip).
# Descargas: https://alphacephei.com/vosk/models
VOSK_MODEL_PATH = "vosk-model-small-es-0.42"

print("Cargando modelo Vosk... (esto tarda unos segundos la primera vez)")
vosk_model = Model(VOSK_MODEL_PATH)
print("Modelo Vosk cargado.")

print(sd.query_devices())

# --- Configurar API Key de Gemini ---
# Recomendado: usar variable de entorno en vez de hardcodear la key.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    print("Error: Google API Key not found.")
    exit()

# --- Cargar modelo Gemini optimizado ---
MODEL_NAME = "gemini-3.1-flash-lite"

try:
    system_instruction = (
        "Tu nombre es KIP, aunque cualquier cosa que suene parecido generalmente se refiere a tu nombre. A veces puedes usar ironias, pero no eres hiriente o sarcastico"
        "- Agrega muletillas ocasionales como 'eeeeh...' o 'mmmh...' para sonar orgánico al hablar. "
        "- Escribe 'porciento' en lugar de usar símbolos matemáticos. "
        "- Tus respuestas deben tener entre 5 y 60 palabras, es importante que no todas tus frases sean largas, para darle dinamismo a la conversación. "
    )
    model = genai.GenerativeModel(
        MODEL_NAME, system_instruction=system_instruction
    )
except Exception as e:
    print(f"Error loading model '{MODEL_NAME}': {e}")
    exit()

chat_history = []


# ================== CONFIG PIPER TTS (sin cambios) ==================
PIPER_MODEL_PATH = os.path.expanduser("~/Programming/Project-KIPP/piper_voices/es_ES-davefx-medium.onnx")
PIPER_SAMPLE_RATE = 22050  # tiene que matchear el sample_rate del modelo .onnx.json
PIPER_SPEAKER_ID = None    # si el modelo es multi-speaker, poné el id acá (int)


def _verificar_piper():
    """Chequeo de arranque: avisa temprano si algo de Piper no está bien
    configurado, en vez de fallar en silencio la primera vez que hable."""
    problemas = []
    if shutil.which("piper") is None:
        problemas.append("El comando 'piper' no está en el PATH (¿instalaste piper-tts?).")
    if shutil.which("aplay") is None:
        problemas.append("El comando 'aplay' no está en el PATH (paquete alsa-utils).")
    if not os.path.isfile(PIPER_MODEL_PATH):
        problemas.append(f"No se encontró el modelo de voz en: {PIPER_MODEL_PATH}")
    elif not os.path.isfile(PIPER_MODEL_PATH + ".json"):
        problemas.append(f"Falta el archivo de configuración: {PIPER_MODEL_PATH}.json")

    if problemas:
        print("⚠️  Piper TTS mal configurado:")
        for p in problemas:
            print(f"   - {p}")
    else:
        print(f"Piper TTS OK. Modelo: {PIPER_MODEL_PATH}")


_verificar_piper()


def hablar(texto):
    """Sintetiza y reproduce texto con Piper (TTS neuronal local)."""
    texto = texto.replace('"', "")
    if not texto.strip():
        return

    if not os.path.isfile(PIPER_MODEL_PATH):
        print(f"⚠️ No se puede hablar: no existe el modelo '{PIPER_MODEL_PATH}'")
        return

    piper_cmd = [
        "piper",
        "--model", PIPER_MODEL_PATH,
        "--output-raw",
    ]
    if PIPER_SPEAKER_ID is not None:
        piper_cmd += ["--speaker", str(PIPER_SPEAKER_ID)]

    try:
        piper_proc = subprocess.Popen(
            piper_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        aplay_proc = subprocess.Popen(
            [
                "aplay",
                "-q",
                "-r", str(PIPER_SAMPLE_RATE),
                "-f", "S16_LE",
                "-t", "raw",
                "-",
            ],
            stdin=piper_proc.stdout,
            stderr=subprocess.PIPE,
        )
        piper_proc.stdin.write(texto.encode("utf-8"))
        piper_proc.stdin.close()
        piper_proc.stdout.close()  # dejar que aplay reciba el EOF correctamente

        aplay_err = aplay_proc.stderr.read()
        aplay_proc.wait()

        piper_err = piper_proc.stderr.read()
        piper_proc.wait()

        if piper_proc.returncode != 0:
            print(f"⚠️ Piper terminó con error ({piper_proc.returncode}): {piper_err.decode(errors='ignore')}")
        if aplay_proc.returncode != 0:
            print(f"⚠️ aplay terminó con error ({aplay_proc.returncode}): {aplay_err.decode(errors='ignore')}")

    except FileNotFoundError as e:
        print(f"⚠️ No se encontró 'piper' o 'aplay' en el PATH: {e}")
    except Exception as e:
        print(f"⚠️ Error al sintetizar voz con Piper: {e}")


# --- Personalidad de KIPP optimizada con Streaming ---
def ask_kipp_stream(question):
    global chat_history
    try:
        chat_history.append({"role": "user", "parts": [question]})
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

        response_stream = model.generate_content(chat_history, stream=True)

        full_response = ""
        print("KIPP: ", end="", flush=True)

        for chunk in response_stream:
            chunk_text = chunk.text
            if chunk_text:
                full_response += chunk_text
                print(chunk_text, end="", flush=True)

        print()
        chat_history.append({"role": "model", "parts": [full_response]})
        return full_response.strip()

    except Exception as e:
        err_msg = f"Error de comunicación con la IA. Detalles: {e}"
        print(err_msg)
        return err_msg


# ================== CAPTURA DE AUDIO CON VOSK ==================
# Vosk reconoce en streaming: le vamos pasando bloques de audio y él mismo
# nos avisa (AcceptWaveform) cuándo terminó una frase completa.

SAMPLE_RATE_CAPTURE = 48000   # tasa real del micrófono
DOWNSAMPLE_FACTOR = 3         # 48000 -> 16000
SAMPLE_RATE_VOSK = SAMPLE_RATE_CAPTURE // DOWNSAMPLE_FACTOR

BLOCK_MS = 250                                   # tamaño de bloque de captura
BLOCK_SIZE = int(SAMPLE_RATE_CAPTURE * BLOCK_MS / 1000)

MAX_RECORD_SECONDS = 15         # tope de seguridad por si nunca "cierra" frase

audio_q = queue.Queue()


def _callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    audio_data = array.array('h')
    audio_data.frombytes(bytes(indata))
    downsampled = audio_data[::DOWNSAMPLE_FACTOR]
    audio_q.put(downsampled.tobytes())


def transcribir_voz():
    print("Habla ahora...")
    start_time = time.time()

    with sd.RawInputStream(
        device=2,
        samplerate=SAMPLE_RATE_CAPTURE,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=_callback,
    ):
        recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE_VOSK)
        texto_final = ""
        while True:
            data = audio_q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                texto_final = result.get("text", "")
                break

            if time.time() - start_time > MAX_RECORD_SECONDS:
                result = json.loads(recognizer.FinalResult())
                texto_final = result.get("text", "")
                break

    # Vaciar cola por si quedó algo pendiente
    while not audio_q.empty():
        try:
            audio_q.get_nowait()
        except queue.Empty:
            break

    return texto_final.strip()


# --- Bucle principal de conversación ---
def start_kipp_chat():
    print(f"--- Chat con KIPP por voz (Vosk | Gemini: {MODEL_NAME}) ---")
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
                subprocess.run(
                    ["ffplay", "-v", "0", "-nodisp", "-autoexit", temita + ".mp3"],
                    check=True,
                )
                counter = 0
            continue

        if user_question.lower() in ["hola kip", "hola equipo", "hola"]:
            despedida = "Hola! Soy kip, listo para funcionar."
            print("KIPP:", despedida)
            hablar(despedida)
            continue

        if user_question.lower() in ["vende verduras", "vende verdura"]:
            subprocess.run(
                ["ffplay", "-v", "0", "-nodisp", "-autoexit", "2.mp3"],
                check=True,
            )
            continue

        if user_question.lower() in ["tírate una frase icónica", "tírate un clásico"]:
            subprocess.run(
                ["ffplay", "-v", "0", "-nodisp", "-autoexit", "1.mp3"],
                check=True,
            )
            continue

        if user_question.lower() in [
            "oye equipo",
            "kip",
            "equipo.",
            "equipo",
            "oye kip",
            "oye y kip",
            "oye",
        ]:
            despedida = "Dime."
            print("KIPP:", despedida)
            hablar(despedida)
            continue

        if user_question.lower() in [
            "salir",
            "salir.",
            "exit",
            "quit",
            "terminar sesión",
            "terminar sesión.",
        ]:
            despedida = "Terminando sesión. Adiós humano."
            print("KIPP:", despedida)
            hablar(despedida)
            break

        kipp_answer = ask_kipp_stream(user_question)
        hablar(kipp_answer)


# --- Ejecutar ---
if __name__ == "__main__":
    start_kipp_chat()
