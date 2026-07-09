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
from faster_whisper import WhisperModel
import sounddevice as sd
import google.generativeai as genai

# ================== CONFIG WHISPER ==================
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
        "- Tus respuestas deben tener entre 5 y 60 palabras, es importante que la mayoria de tus respuestas sean cortas no mas de 10 palabras, para dar una conversacion fluida y de vez en cuando, cuando la ocasion lo amerite usar respuestas largas cuando necesites explicar algo. "
    )
    model = genai.GenerativeModel(
        MODEL_NAME, system_instruction=system_instruction
    )
except Exception as e:
    print(f"Error loading model '{MODEL_NAME}': {e}")
    exit()

chat_history = []


# ================== CONFIG PIPER TTS ==================
PIPER_MODEL_PATH = os.path.expanduser("~/Programming/Project-KIPP/piper_voices/es_ES-davefx-medium.onnx")
PIPER_SAMPLE_RATE = 22050  
PIPER_SPEAKER_ID = None    


def _verificar_piper():
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

# --- NUEVO: Variables globales para controlar la interrupción ---
active_piper_proc = None
active_aplay_proc = None

def detener_hablar():
    """Mata los procesos de Piper y aplay si están activos."""
    global active_piper_proc, active_aplay_proc
    try:
        if active_aplay_proc and active_aplay_proc.poll() is None:
            active_aplay_proc.terminate()
        if active_piper_proc and active_piper_proc.poll() is None:
            active_piper_proc.terminate()
    except Exception as e:
        pass


def hablar(texto):
    """Sintetiza y reproduce texto con Piper. Ahora permite ser interrumpido."""
    global active_piper_proc, active_aplay_proc
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
        active_piper_proc = subprocess.Popen(
            piper_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        active_aplay_proc = subprocess.Popen(
            [
                "aplay",
                "-q",
                "-r", str(PIPER_SAMPLE_RATE),
                "-f", "S16_LE",
                "-t", "raw",
                "-",
            ],
            stdin=active_piper_proc.stdout,
            stderr=subprocess.PIPE,
        )
        active_piper_proc.stdin.write(texto.encode("utf-8"))
        active_piper_proc.stdin.close()
        active_piper_proc.stdout.close() 

        # Esperamos a que termine (si se llama a terminate() desde otro hilo, esto se desbloquea rápido)
        active_aplay_proc.wait()
        active_piper_proc.wait()

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


# ================== CAPTURA DE AUDIO CON VAD POR ENERGÍA ==================
SAMPLE_RATE_CAPTURE = 48000   
DOWNSAMPLE_FACTOR = 3         
SAMPLE_RATE_WHISPER = SAMPLE_RATE_CAPTURE // DOWNSAMPLE_FACTOR

BLOCK_MS = 100                                                
BLOCK_SIZE = int(SAMPLE_RATE_CAPTURE * BLOCK_MS / 1000)

SILENCE_RMS_THRESHOLD = 300     # ajustar según tu micrófono/ruido ambiente
SILENCE_HANGOVER_MS = 100       # cuánto silencio esperar antes de cortar
MAX_RECORD_SECONDS = 15         # tope de seguridad
MIN_SPEECH_MS = 250             # mínimo de voz detectada para no descartar como ruido

audio_q = queue.Queue()


def _callback(indata, frames, time_info, status):
    if status:
        pass # print("⚠️", status) evitamos spam visual durante la grabación
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
    
    global active_aplay_proc

    with sd.RawInputStream(
        device=2, # <-- Asegúrate de que este ID sea el de tu micrófono
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
                # --- NUEVO: SISTEMA DE INTERRUPCIÓN ---
                # Si el volumen sube y KIPP está hablando, lo silenciamos
                if active_aplay_proc and active_aplay_proc.poll() is None:
                    print("\n[!] Interrupción detectada. Deteniendo voz de KIPP...")
                    detener_hablar()
                    
                    # Reiniciamos la grabación descartando el audio anterior
                    # para que la interrupción en sí no ensucie el nuevo prompt
                    frames = [chunk]
                    speech_ms = BLOCK_MS
                    silence_ms = 0
                    started_speaking = True
                    start_time = time.time()
                    continue
                # ----------------------------------------

                started_speaking = True
                speech_ms += BLOCK_MS
                silence_ms = 0
            else:
                if started_speaking:
                    silence_ms += BLOCK_MS

            if started_speaking and silence_ms >= SILENCE_HANGOVER_MS:
                break

            if time.time() - start_time > MAX_RECORD_SECONDS:
                break

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
        beam_size=1,          
        vad_filter=True,      
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
        if user_question:
            print(f"Tú (voz): {user_question}")

        if not user_question.strip():
            print("KIPP: Entrada vacía. Intenta de nuevo.")
            counter += 1
            if counter == 5:
                temita = str(random.randint(1, 3))
                # Lanzamos el audio en background para que no bloquee (opcional)
                subprocess.Popen(["ffplay", "-v", "0", "-nodisp", "-autoexit", temita + ".mp3"])
                counter = 0
            continue

        if user_question.lower() in ["hola kip", "hola equipo", "hola"]:
            despedida = "Hola! Soy kip, listo para funcionar."
            print("KIPP:", despedida)
            threading.Thread(target=hablar, args=(despedida,), daemon=True).start()
            continue

        if user_question.lower() in ["vende verduras", "vende verdura"]:
            subprocess.Popen(["ffplay", "-v", "0", "-nodisp", "-autoexit", "2.mp3"])
            continue

        if user_question.lower() in ["tírate una frase icónica", "tírate un clásico"]:
            subprocess.Popen(["ffplay", "-v", "0", "-nodisp", "-autoexit", "1.mp3"])
            continue

        if user_question.lower() in [
            "oye equipo", "kip", "equipo.", "equipo", "oye kip", "oye y kip", "oye",
        ]:
            despedida = "Dime."
            print("KIPP:", despedida)
            threading.Thread(target=hablar, args=(despedida,), daemon=True).start()
            continue

        if user_question.lower() in [
            "salir", "exit", "quit", "terminar sesión", "terminar sesión.",
        ]:
            despedida = "Terminando sesión. Adiós humano."
            print("KIPP:", despedida)
            hablar(despedida) # Dejamos este síncrono para que termine de hablar antes de cerrar el script
            break

        kipp_answer = ask_kipp_stream(user_question)
        
        # --- NUEVO: Ejecutar hablar() en un hilo para no bloquear el micrófono ---
        threading.Thread(target=hablar, args=(kipp_answer,), daemon=True).start()


if __name__ == "__main__":
    start_kipp_chat()
