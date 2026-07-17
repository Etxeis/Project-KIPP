import json
import os
import queue
import random
import subprocess
import threading
import time
import array
import re
import numpy as np
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import google.generativeai as genai

# ================== CONFIG VOSK ==================
print("Cargando modelo Vosk...")
vosk_model = Model("vosk-model-small-es-0.42")
print("Modelo Vosk cargado.")
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
        "Tu nombre es KIP, aunque cualquier cosa que suene parecido generalmente se refiere a tu nombre. A veces puedes usar ironias, pero no eres hiriente o sarcastico "
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


# ================== TTS EN BACKGROUND (NO BLOQUEANTE) ==================
tts_queue = queue.Queue()

def _hablar_bloqueante(texto):
    texto = texto.replace('"', "").strip()
    if not texto:
        return
    try:
        # Se cambió a "Mateo" como en el script de Whisper (puedes volver a "Alan" si prefieres)
        subprocess.run(
            ["rhvoice.test", "-p", "Mateo"],
            input=texto.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("⚠️ No se encontró 'rhvoice.test' en el PATH.")

def _tts_worker():
    while True:
        texto = tts_queue.get()
        try:
            if texto is None:
                break
            _hablar_bloqueante(texto)
        finally:
            tts_queue.task_done()

threading.Thread(target=_tts_worker, daemon=True).start()

def hablar(texto):
    """Encola una frase para hablar. No bloquea: retorna al instante."""
    tts_queue.put(texto)

def esperar_habla():
    """Bloquea hasta que se haya terminado de decir todo lo encolado hasta ahora."""
    tts_queue.join()

_FIN_DE_ORACION_RE = re.compile(r"(?<=[\.\!\?\;\:])\s+")


# ================== GENERACIÓN DE RESPUESTA STREAMING ==================
def ask_kipp_stream(question):
    global chat_history
    try:
        chat_history.append({"role": "user", "parts": [question]})
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

        response_stream = model.generate_content(chat_history, stream=True)

        full_response = ""
        buffer = ""
        print("KIPP: ", end="", flush=True)

        for chunk in response_stream:
            chunk_text = chunk.text
            if not chunk_text:
                continue

            full_response += chunk_text
            buffer += chunk_text
            print(chunk_text, end="", flush=True)

            # Apenas se cierra una oración, la mandamos a hablar
            partes = _FIN_DE_ORACION_RE.split(buffer)
            if len(partes) > 1:
                for oracion in partes[:-1]:
                    if oracion.strip():
                        hablar(oracion.strip())
                buffer = partes[-1]

        # Restante sin puntuación
        if buffer.strip():
            hablar(buffer.strip())

        print()
        chat_history.append({"role": "model", "parts": [full_response]})
        return full_response.strip()

    except Exception as e:
        err_msg = f"Error de comunicación con la IA. Detalles: {e}"
        print(err_msg)
        hablar(err_msg)
        return err_msg


# ================== CAPTURA DE AUDIO + VAD (MULETILLAS) CON VOSK ==================
SAMPLE_RATE_CAPTURE = 48000   
DOWNSAMPLE_FACTOR = 3         # 48000 -> 16000
BLOCK_MS = 100                
BLOCK_SIZE = int(SAMPLE_RATE_CAPTURE * BLOCK_MS / 1000)

SILENCE_RMS_THRESHOLD = 300   
MAX_RECORD_SECONDS = 15       
MIN_SPEECH_MS = 250           
SILENCE_SOFT_MS = 600
SILENCE_HARD_MS = 1600
MAX_EXTENSIONS = 3          

PALABRAS_DE_PAUSA = {
    "mmm", "mm", "eh", "ehh", "em", "emm", "este", "esteee",
    "o sea", "osea", "digo", "bueno", "pues", "ehm", "aja", "ajá",
}
CONECTORES_INCOMPLETOS = {
    "y", "pero", "porque", "que", "de", "para", "con", "sin",
    "a", "en", "o", "entonces", "aunque", "como", "cuando",
}

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

def _texto_sugiere_continuacion(texto):
    if not texto:
        return False
    texto = texto.strip().lower()
    texto = texto.rstrip(".,!?¡¿…")
    if not texto:
        return False

    for muletilla in PALABRAS_DE_PAUSA:
        if " " in muletilla and texto.endswith(muletilla):
            return True

    palabras = re.findall(r"[\wáéíóúñ]+", texto)
    if not palabras:
        return False
    return palabras[-1] in PALABRAS_DE_PAUSA or palabras[-1] in CONECTORES_INCOMPLETOS


def transcribir_voz():
    print("Habla ahora...")
    silence_ms = 0
    speech_ms = 0
    started_speaking = False
    start_time = time.time()
    extensiones = 0
    checkpoint_hecho = False

    recognizer = KaldiRecognizer(vosk_model, 16000)
    texto_final_acumulado = ""

    with sd.RawInputStream(
        device=2,  # Ajusta al índice correcto si es necesario
        samplerate=SAMPLE_RATE_CAPTURE,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=_callback,
    ):
        while True:
            chunk = audio_q.get()
            level = _rms(chunk)

            # Vamos alimentando a Vosk en tiempo real
            if recognizer.AcceptWaveform(chunk):
                # Vosk detectó un fin de frase interno
                res = json.loads(recognizer.Result())
                txt = res.get("text", "").strip()
                if txt:
                    texto_final_acumulado += txt + " "

            # Lógica de VAD por energía
            if level >= SILENCE_RMS_THRESHOLD:
                started_speaking = True
                speech_ms += BLOCK_MS
                silence_ms = 0
                checkpoint_hecho = False
            else:
                if started_speaking:
                    silence_ms += BLOCK_MS

            if started_speaking:
                # Corte duro
                if silence_ms >= SILENCE_HARD_MS:
                    break

                # Corte suave (evalúa muletillas)
                if silence_ms >= SILENCE_SOFT_MS and not checkpoint_hecho:
                    checkpoint_hecho = True
                    
                    # Obtenemos lo que Vosk esté entendiendo hasta este instante
                    res_parcial = json.loads(recognizer.PartialResult())
                    txt_parcial = res_parcial.get("partial", "").strip()
                    
                    texto_a_evaluar = txt_parcial if txt_parcial else texto_final_acumulado

                    if _texto_sugiere_continuacion(texto_a_evaluar) and extensiones < MAX_EXTENSIONS:
                        print(f"  (pausa detectada tras: '{texto_a_evaluar}', dando más tiempo...)")
                        extensiones += 1
                        silence_ms = 0
                        continue
                    else:
                        break

            # Tope de seguridad
            if time.time() - start_time > MAX_RECORD_SECONDS:
                break

    # Vaciar cola residual
    while not audio_q.empty():
        try:
            audio_q.get_nowait()
        except queue.Empty:
            break

    if not started_speaking or speech_ms < MIN_SPEECH_MS:
        return ""

    # Extraer el resultado final remanente de Vosk
    res_final = json.loads(recognizer.FinalResult())
    txt_final = res_final.get("text", "").strip()
    if txt_final:
        texto_final_acumulado += txt_final

    return texto_final_acumulado.strip()


# ================== BUCLE PRINCIPAL ==================
def start_kipp_chat():
    print(f"--- Chat con KIPP por voz (Vosk Streaming | Gemini: {MODEL_NAME}) ---")
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
            esperar_habla()
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
            "oye equipo", "kip", "equipo", "oye kip", "oye y kip", "oye"
        ]:
            despedida = "Dime."
            print("KIPP:", despedida)
            hablar(despedida)
            esperar_habla()
            continue

        if user_question.lower() in [
            "salir", "salir.", "exit", "quit", 
            "terminar sesión", "terminar sesión.", "terminar sesión", 
            "terminar sesión.", "terminar sesión."
        ]:
            despedida = "Terminando sesión. Adiós humano."
            print("KIPP:", despedida)
            hablar(despedida)
            esperar_habla()
            break

        # ask_kipp_stream procesa e inserta a la cola de voz concurrentemente
        ask_kipp_stream(user_question)
        esperar_habla()


if __name__ == "__main__":
    start_kipp_chat()
