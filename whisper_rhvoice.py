import json
import os
import queue
import random
import subprocess
import threading
import time
import array
import numpy as np
from faster_whisper import WhisperModel
import sounddevice as sd
import google.generativeai as genai

# ================== CONFIG WHISPER ==================
# Tamaños disponibles: tiny, base, small, medium, large-v3
# "small" es un buen balance velocidad/precisión para español.
# device="cuda" si tenés GPU NVIDIA, si no "cpu".
# compute_type="int8" es lo más rápido en CPU. En GPU usa "float16".
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"          # "cuda" si tenés GPU
WHISPER_COMPUTE_TYPE = "int8"   # "int8" (cpu rápido) / "float16" (gpu) / "int8_float16"

print("Cargando modelo Whisper... (esto tarda unos segundos la primera vez)")
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)
print("Modelo Whisper cargado.")

print(sd.query_devices())

# --- Configurar API Key de Gemini ---
try:
    genai.configure(api_key="")  # <-- Coloca tu clave aquí
except KeyError:
    print("Error: Google API Key not found.")
    exit()

# --- Cargar modelo Gemini optimizado ---
MODEL_NAME = "gemini-3.1-flash-lite"

try:
    system_instruction = (
        "Tu nombre es KIP, aunque cualquier cosa que suene parecido generalmente se refiere a tu nombre. A veces puedes usar ironias, pero no eres hiriente o sarcastico"
        "- Escribe 'porciento' en lugar de usar símbolos matemáticos. "
        "- Tus respuestas deben tener entre 5 y 80 palabras, es importante que la mayoria de tus respuestas sean cortas no mas de 10 palabras, para dar una conversacion fluida y de vez en cuando, cuando la ocasion lo amerite usar respuestas largas cuando necesites explicar algo. "
    )
    model = genai.GenerativeModel(
        MODEL_NAME, system_instruction=system_instruction
    )
except Exception as e:
    print(f"Error loading model '{MODEL_NAME}': {e}")
    exit()

chat_history = []


# --- Función de texto a voz usando rhvoice-test ---
def hablar(texto):
    texto = texto.replace('"', "")
    os.system(f'echo "{texto}"|rhvoice.test -p "Mateo"')


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


# ================== CAPTURA DE AUDIO CON VAD POR ENERGÍA ==================
# En vez de reconocer en streaming (como hacía Vosk), grabamos hasta detectar
# silencio y recién ahí transcribimos todo el fragmento con Whisper.
# Esto es lo que hace que sea rápido: Whisper procesa un audio corto ya
# "cerrado" en vez de tener que decodificar en vivo.

SAMPLE_RATE_CAPTURE = 48000   # tasa real del micrófono
DOWNSAMPLE_FACTOR = 3         # 48000 -> 16000
SAMPLE_RATE_WHISPER = SAMPLE_RATE_CAPTURE // DOWNSAMPLE_FACTOR

BLOCK_MS = 100                                  # tamaño de bloque de análisis
BLOCK_SIZE = int(SAMPLE_RATE_CAPTURE * BLOCK_MS / 1000)

SILENCE_RMS_THRESHOLD = 300     # ajustar según tu micrófono/ruido ambiente
SILENCE_HANGOVER_MS = 100       # cuánto silencio esperar antes de cortar
MAX_RECORD_SECONDS = 15         # tope de seguridad
MIN_SPEECH_MS = 250             # mínimo de voz detectada para no descartar como ruido

audio_q = queue.Queue()


def _callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    audio_data = array.array('h')
    audio_data.frombytes(bytes(indata))
    downsampled = audio_data[::DOWNSAMPLE_FACTOR]
    audio_q.put(downsampled.tobytes())


def _rms(pcm_bytes):
    if not pcm_bytes:
        return 0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0
    return float(np.sqrt(np.mean(samples ** 2)))


def transcribir_voz():
    print("Habla ahora...")
    frames = []
    silence_ms = 0
    speech_ms = 0
    started_speaking = False
    start_time = time.time()

    with sd.RawInputStream(
        device=2,
        samplerate=SAMPLE_RATE_CAPTURE,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=_callback,
    ):
        while True:
            chunk = audio_q.get()
            frames.append(chunk)

            level = _rms(chunk)

            if level >= SILENCE_RMS_THRESHOLD:
                started_speaking = True
                speech_ms += BLOCK_MS
                silence_ms = 0
            else:
                if started_speaking:
                    silence_ms += BLOCK_MS

            # Cortar si hubo voz y luego suficiente silencio
            if started_speaking and silence_ms >= SILENCE_HANGOVER_MS:
                break

            # Tope de seguridad para no grabar infinito
            if time.time() - start_time > MAX_RECORD_SECONDS:
                break

    # Vaciar cola por si quedó algo pendiente
    while not audio_q.empty():
        try:
            audio_q.get_nowait()
        except queue.Empty:
            break

    if not started_speaking or speech_ms < MIN_SPEECH_MS:
        return ""

    raw_pcm = b"".join(frames)
    audio_np = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0

    segments, _info = whisper_model.transcribe(
        audio_np,
        language="es",
        beam_size=1,          # beam_size=1 = mucho más rápido (greedy)
        vad_filter=True,      # recorta silencios internos del propio Whisper
        condition_on_previous_text=False,
    )

    texto_final = "".join(seg.text for seg in segments).strip()
    return texto_final


# --- Bucle principal de conversación ---
def start_kipp_chat():
    print(f"--- Chat con KIPP por voz (Whisper: {WHISPER_MODEL_SIZE} | Gemini: {MODEL_NAME}) ---")
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
            "Salir.",
            "exit",
            "quit",
            "terminar sesión",
            "terminar sesión.",
            "Terminar sesión",
            "Terminar Sesión",
            "Terminar sesión.",
            "Terminar Sesión.",
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
