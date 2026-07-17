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


# --- Texto a voz con cola en background (para respuesta casi inmediata) ---
# Antes: os.system(f'echo "{texto}"|rhvoice.test ...') abre una shell nueva
# y hace un pipe cada vez que se llama, y además BLOQUEA todo el programa
# mientras habla. Ahora:
#   - hablar() solo encola el texto y retorna al instante (no bloquea).
#   - un hilo en background va sacando frases de la cola y las reproduce
#     con subprocess (sin shell de por medio, pasando el texto por stdin).
#   - esto permite ir hablando oración por oración apenas Gemini las genera,
#     en vez de esperar la respuesta completa.
tts_queue = queue.Queue()


def _hablar_bloqueante(texto):
    texto = texto.replace('"', "").strip()
    if not texto:
        return
    try:
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


# Separa por fin de oración (. ! ? ; :) seguido de espacio, para poder
# hablar cada oración apenas está lista sin esperar el resto del texto.
_FIN_DE_ORACION_RE = re.compile(r"(?<=[\.\!\?\;\:])\s+")


# --- Personalidad de KIPP optimizada con Streaming ---
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

            # Apenas se cierra una oración dentro del buffer, la hablamos YA
            # (no esperamos a que termine toda la respuesta de Gemini).
            partes = _FIN_DE_ORACION_RE.split(buffer)
            if len(partes) > 1:
                for oracion in partes[:-1]:
                    if oracion.strip():
                        hablar(oracion.strip())
                buffer = partes[-1]

        # lo que haya quedado sin puntuación de cierre también se dice
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
MAX_RECORD_SECONDS = 15         # tope de seguridad
MIN_SPEECH_MS = 250             # mínimo de voz detectada para no descartar como ruido

# --- Pausas "semánticas" ---
# Antes se cortaba apenas había SILENCE_HANGOVER_MS de silencio (100ms), lo que
# hacía que cualquier "mmm..." o pausa para pensar cortara la grabación.
# Ahora manejamos dos umbrales:
#   - SILENCE_SOFT_MS: al llegar acá, no cortamos todavía. Hacemos un chequeo
#     rápido (transcripción parcial) para ver si la frase "suena incompleta".
#   - SILENCE_HARD_MS: tope duro. Pase lo que pase, si el silencio llega acá,
#     se corta (para no quedar escuchando para siempre).
SILENCE_SOFT_MS = 600
SILENCE_HARD_MS = 1600
MAX_EXTENSIONS = 3          # cuántas veces se puede "dar más tiempo" por pausa

# Muletillas / palabras de duda: si la frase termina en una de estas,
# es una señal fuerte de que la persona sigue pensando y va a continuar.
PALABRAS_DE_PAUSA = {
    "mmm", "mm", "eh", "ehh", "em", "emm", "este", "esteee",
    "o sea", "osea", "digo", "bueno", "pues", "ehm", "aja", "ajá",
}

# Conectores que, si quedan "colgados" al final, sugieren que la oración
# no terminó todavía (señal más débil, pero útil).
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


def _frames_a_audio_np(frames):
    raw_pcm = b"".join(frames)
    return np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _transcribir_rapido(frames):
    """Transcripción parcial y barata, solo para decidir si seguir escuchando."""
    audio_np = _frames_a_audio_np(frames)
    if audio_np.size == 0:
        return ""
    try:
        segments, _info = whisper_model.transcribe(
            audio_np,
            language="es",
            beam_size=1,
            vad_filter=False,             # ya sabemos que hay voz, no hace falta
            condition_on_previous_text=False,
        )
        return "".join(seg.text for seg in segments).strip()
    except Exception:
        return ""


def _texto_sugiere_continuacion(texto):
    """True si la frase parece cortada a la mitad (muletilla o conector colgado)."""
    if not texto:
        return False
    texto = texto.strip().lower()
    texto = texto.rstrip(".,!?¡¿…")
    if not texto:
        return False

    # muletilla de dos palabras tipo "o sea"
    for muletilla in PALABRAS_DE_PAUSA:
        if " " in muletilla and texto.endswith(muletilla):
            return True

    palabras = re.findall(r"[\wáéíóúñ]+", texto)
    if not palabras:
        return False
    ultima_palabra = palabras[-1]

    return ultima_palabra in PALABRAS_DE_PAUSA or ultima_palabra in CONECTORES_INCOMPLETOS


def transcribir_voz():
    print("Habla ahora...")
    frames = []
    silence_ms = 0
    speech_ms = 0
    started_speaking = False
    start_time = time.time()

    extensiones = 0
    checkpoint_hecho = False
    ultimo_bloque_con_voz = -1  # índice del último bloque con voz detectada

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
                checkpoint_hecho = False  # volvió a hablar, se resetea el chequeo
                ultimo_bloque_con_voz = len(frames) - 1
            else:
                if started_speaking:
                    silence_ms += BLOCK_MS

            if started_speaking:
                # Tope duro: no importa si "suena incompleta", cortamos igual.
                if silence_ms >= SILENCE_HARD_MS:
                    break

                # Silencio blando: chequeo semántico antes de decidir cortar.
                if silence_ms >= SILENCE_SOFT_MS and not checkpoint_hecho:
                    checkpoint_hecho = True

                    if extensiones < MAX_EXTENSIONS:
                        texto_parcial = _transcribir_rapido(frames)
                        if _texto_sugiere_continuacion(texto_parcial):
                            print(f"  (pausa detectada tras: '{texto_parcial}', dando más tiempo...)")
                            extensiones += 1
                            silence_ms = 0
                            continue

                    # No hay señal de continuación (o ya no quedan extensiones)
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

    # Recortamos la cola de silencio (podés haber esperado hasta SILENCE_HARD_MS
    # antes de cortar) dejando solo un pequeño colchón. Menos audio == Whisper
    # más rápido, y evita que la respuesta se demore por segundos de silencio
    # que no aportan nada a la transcripción.
    PADDING_BLOQUES = 3  # ~300ms de colchón después de la última voz detectada
    if ultimo_bloque_con_voz >= 0:
        frames = frames[: ultimo_bloque_con_voz + 1 + PADDING_BLOQUES]

    audio_np = _frames_a_audio_np(frames)

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
            esperar_habla()
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
            esperar_habla()
            break

        # ask_kipp_stream ya va encolando y hablando cada oración a medida
        # que Gemini la genera, así que no hace falta volver a llamar hablar()
        # acá con la respuesta completa (eso duplicaría el audio).
        ask_kipp_stream(user_question)
        esperar_habla()


# --- Ejecutar ---
if __name__ == "__main__":
    start_kipp_chat()
