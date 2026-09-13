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
import requests
from vosk import Model, KaldiRecognizer
import sounddevice as sd


# ============================================================
# ================== CONFIGURACIÓN OLLAMA ====================
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:1.7b"

SYSTEM_INSTRUCTION = """
Eres KIP, un asistente de voz.

Tu nombre es KIP, aunque cualquier cosa que suene parecido
generalmente se refiere a tu nombre.

Responde siempre en español.

Sé directo, natural y breve.

La mayoría de tus respuestas deben tener menos de 10 palabras.
Puedes utilizar hasta 80 palabras cuando sea necesario explicar algo.

No repitas la pregunta del usuario.
No describas tus pensamientos internos.
No uses Markdown.
No uses listas salvo que sean realmente necesarias.

Puedes utilizar humor e ironía ligera, pero nunca seas hiriente
ni excesivamente sarcástico.

Escribe "porciento" en lugar de utilizar el símbolo "%".
"""

# Historial de conversación
chat_history = []


# ============================================================
# ================== COMPROBAR OLLAMA ========================
# ============================================================

def comprobar_ollama():
    print("Comprobando conexión con Ollama...")

    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )

        response.raise_for_status()

        modelos = response.json().get("models", [])

        nombres = [
            modelo.get("name", "")
            for modelo in modelos
        ]

        if MODEL_NAME not in nombres:
            print()
            print(f"⚠️ No se encontró el modelo '{MODEL_NAME}'.")
            print()
            print("Modelos disponibles:")

            for nombre in nombres:
                print(f"  - {nombre}")

            print()
            print(f"Instálalo con:")
            print(f"  ollama pull {MODEL_NAME}")
            print()

            return False

        print(f"Ollama conectado correctamente.")
        print(f"Modelo encontrado: {MODEL_NAME}")
        print()

        return True

    except requests.exceptions.ConnectionError:
        print()
        print("⚠️ No se pudo conectar con Ollama.")
        print()
        print("Asegúrate de que Ollama esté ejecutándose:")
        print("  ollama serve")
        print()

        return False

    except Exception as e:
        print(f"⚠️ Error comprobando Ollama: {e}")
        return False


if not comprobar_ollama():
    exit()


# ============================================================
# ================== CONFIG VOSK =============================
# ============================================================

print("Cargando modelo Vosk...")

vosk_model = Model("vosk-model-small-es-0.42")

print("Modelo Vosk cargado.")

print(sd.query_devices())


# ============================================================
# ================== TTS EN BACKGROUND =======================
# ============================================================

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


threading.Thread(
    target=_tts_worker,
    daemon=True
).start()


def hablar(texto):
    """
    Encola una frase para hablar.
    No bloquea la ejecución.
    """

    tts_queue.put(texto)


def esperar_habla():
    """
    Espera hasta que se haya terminado de reproducir
    todo el contenido actualmente en la cola.
    """

    tts_queue.join()


# Detecta finales de oración para enviar frases al TTS
_FIN_DE_ORACION_RE = re.compile(
    r"(?<=[\.\!\?\;\:])\s+"
)


# ============================================================
# ================== OLLAMA STREAMING ========================
# ============================================================

def ask_kipp_stream(question):

    global chat_history

    try:

        # ----------------------------------------------------
        # Agregar pregunta del usuario al historial
        # ----------------------------------------------------

        chat_history.append({
            "role": "user",
            "content": question
        })

        # ----------------------------------------------------
        # Mantener historial limitado
        # ----------------------------------------------------

        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

        # ----------------------------------------------------
        # Construir mensajes
        # ----------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION
            }
        ]

        messages.extend(chat_history)

        # ----------------------------------------------------
        # Configuración Ollama
        # ----------------------------------------------------

        payload = {
            "model": MODEL_NAME,

            "messages": messages,

            "stream": True,

            # Desactivar thinking para reducir latencia
            "think": False,

            "options": {
                "temperature": 0.7
            }
        }

        print("KIPP: ", end="", flush=True)

        full_response = ""
        buffer = ""

        # ----------------------------------------------------
        # Solicitud streaming
        # ----------------------------------------------------

        with requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=None
        ) as response:

            response.raise_for_status()

            # ------------------------------------------------
            # Leer respuesta de Ollama
            # ------------------------------------------------

            for line in response.iter_lines():

                if not line:
                    continue

                try:
                    data = json.loads(line)

                except json.JSONDecodeError:
                    continue

                chunk_text = (
                    data
                    .get("message", {})
                    .get("content", "")
                )

                if not chunk_text:
                    continue

                # --------------------------------------------
                # Acumular respuesta
                # --------------------------------------------

                full_response += chunk_text
                buffer += chunk_text

                # Mostrar en terminal
                print(
                    chunk_text,
                    end="",
                    flush=True
                )

                # --------------------------------------------
                # Detectar oraciones completas
                # --------------------------------------------

                partes = _FIN_DE_ORACION_RE.split(buffer)

                if len(partes) > 1:

                    for oracion in partes[:-1]:

                        if oracion.strip():
                            hablar(oracion.strip())

                    buffer = partes[-1]

                # --------------------------------------------
                # Ollama terminó
                # --------------------------------------------

                if data.get("done", False):
                    break

        # ----------------------------------------------------
        # Hablar texto restante
        # ----------------------------------------------------

        if buffer.strip():
            hablar(buffer.strip())

        print()

        # ----------------------------------------------------
        # Guardar respuesta
        # ----------------------------------------------------

        chat_history.append({
            "role": "assistant",
            "content": full_response
        })

        return full_response.strip()

    # ========================================================
    # Errores
    # ========================================================

    except requests.exceptions.ConnectionError:

        err_msg = (
            "No puedo conectarme con Ollama. "
            "Asegúrate de que Ollama esté funcionando."
        )

        print(err_msg)

        hablar(err_msg)

        return err_msg

    except requests.exceptions.Timeout:

        err_msg = (
            "Ollama está tardando demasiado en responder."
        )

        print(err_msg)

        hablar(err_msg)

        return err_msg

    except Exception as e:

        err_msg = f"Error comunicando con Ollama: {e}"

        print(err_msg)

        hablar(err_msg)

        return err_msg


# ============================================================
# ================== CAPTURA DE AUDIO ========================
# ============================================================

SAMPLE_RATE_CAPTURE = 48000

DOWNSAMPLE_FACTOR = 3

BLOCK_MS = 100

BLOCK_SIZE = int(
    SAMPLE_RATE_CAPTURE * BLOCK_MS / 1000
)


# ============================================================
# ================== CONFIGURACIÓN VAD =======================
# ============================================================

SILENCE_RMS_THRESHOLD = 300

MAX_RECORD_SECONDS = 15

MIN_SPEECH_MS = 250

SILENCE_SOFT_MS = 600

SILENCE_HARD_MS = 1600

MAX_EXTENSIONS = 3


# ============================================================
# ================== MULETILLAS ==============================
# ============================================================

PALABRAS_DE_PAUSA = {
    "mmm",
    "mm",
    "eh",
    "ehh",
    "em",
    "emm",
    "este",
    "esteee",
    "o sea",
    "osea",
    "digo",
    "bueno",
    "pues",
    "ehm",
    "aja",
    "ajá",
}


CONECTORES_INCOMPLETOS = {
    "y",
    "pero",
    "porque",
    "que",
    "de",
    "para",
    "con",
    "sin",
    "a",
    "en",
    "o",
    "entonces",
    "aunque",
    "como",
    "cuando",
}


# ============================================================
# ================== COLA DE AUDIO ===========================
# ============================================================

audio_q = queue.Queue()


def _callback(indata, frames, time_info, status):

    if status:
        print("⚠️", status)

    audio_data = array.array('h')

    audio_data.frombytes(
        bytes(indata)
    )

    downsampled = audio_data[
        ::DOWNSAMPLE_FACTOR
    ]

    audio_q.put(
        downsampled.tobytes()
    )


# ============================================================
# ================== CALCULAR RMS ============================
# ============================================================

def _rms(pcm_bytes):

    if not pcm_bytes:
        return 0

    samples = np.frombuffer(
        pcm_bytes,
        dtype=np.int16
    ).astype(np.float32)

    if samples.size == 0:
        return 0

    return float(
        np.sqrt(
            np.mean(samples ** 2)
        )
    )


# ============================================================
# ================== DETECTAR CONTINUACIÓN ===================
# ============================================================

def _texto_sugiere_continuacion(texto):

    if not texto:
        return False

    texto = texto.strip().lower()

    texto = texto.rstrip(
        ".,!?¡¿…"
    )

    if not texto:
        return False

    # Frases que indican que el usuario probablemente
    # todavía no terminó de hablar.

    for muletilla in PALABRAS_DE_PAUSA:

        if " " in muletilla:

            if texto.endswith(muletilla):
                return True

    palabras = re.findall(
        r"[\wáéíóúñ]+",
        texto
    )

    if not palabras:
        return False

    ultima_palabra = palabras[-1]

    return (
        ultima_palabra in PALABRAS_DE_PAUSA
        or
        ultima_palabra in CONECTORES_INCOMPLETOS
    )


# ============================================================
# ================== TRANSCRIPCIÓN DE VOZ ===================
# ============================================================

def transcribir_voz():

    print("Habla ahora...")

    silence_ms = 0

    speech_ms = 0

    started_speaking = False

    start_time = time.time()

    extensiones = 0

    checkpoint_hecho = False

    recognizer = KaldiRecognizer(
        vosk_model,
        16000
    )

    texto_final_acumulado = ""

    # --------------------------------------------------------
    # Abrir micrófono
    # --------------------------------------------------------

    with sd.RawInputStream(

        device=1,

        samplerate=SAMPLE_RATE_CAPTURE,

        blocksize=BLOCK_SIZE,

        dtype="int16",

        channels=1,

        callback=_callback,

    ):

        while True:

            chunk = audio_q.get()

            level = _rms(chunk)

            # ------------------------------------------------
            # Alimentar Vosk
            # ------------------------------------------------

            if recognizer.AcceptWaveform(chunk):

                res = json.loads(
                    recognizer.Result()
                )

                txt = res.get(
                    "text",
                    ""
                ).strip()

                if txt:

                    texto_final_acumulado += (
                        txt + " "
                    )

            # ------------------------------------------------
            # VAD por energía
            # ------------------------------------------------

            if level >= SILENCE_RMS_THRESHOLD:

                started_speaking = True

                speech_ms += BLOCK_MS

                silence_ms = 0

                checkpoint_hecho = False

            else:

                if started_speaking:

                    silence_ms += BLOCK_MS

            # ------------------------------------------------
            # Si ya comenzó a hablar
            # ------------------------------------------------

            if started_speaking:

                # --------------------------------------------
                # Corte duro
                # --------------------------------------------

                if silence_ms >= SILENCE_HARD_MS:
                    break

                # --------------------------------------------
                # Corte suave
                # --------------------------------------------

                if (
                    silence_ms >= SILENCE_SOFT_MS
                    and not checkpoint_hecho
                ):

                    checkpoint_hecho = True

                    # ----------------------------------------
                    # Obtener resultado parcial
                    # ----------------------------------------

                    res_parcial = json.loads(
                        recognizer.PartialResult()
                    )

                    txt_parcial = (
                        res_parcial
                        .get("partial", "")
                        .strip()
                    )

                    texto_a_evaluar = (
                        txt_parcial
                        if txt_parcial
                        else texto_final_acumulado
                    )

                    # ----------------------------------------
                    # Determinar si probablemente continúa
                    # ----------------------------------------

                    if (
                        _texto_sugiere_continuacion(
                            texto_a_evaluar
                        )
                        and extensiones < MAX_EXTENSIONS
                    ):

                        print(
                            "  "
                            f"(pausa detectada tras: "
                            f"'{texto_a_evaluar}', "
                            "dando más tiempo...)"
                        )

                        extensiones += 1

                        silence_ms = 0

                        continue

                    else:

                        break

            # ------------------------------------------------
            # Tope de seguridad
            # ------------------------------------------------

            if (
                time.time() - start_time
                > MAX_RECORD_SECONDS
            ):
                break

    # ========================================================
    # Vaciar cola residual
    # ========================================================

    while not audio_q.empty():

        try:

            audio_q.get_nowait()

        except queue.Empty:

            break

    # ========================================================
    # Validar duración mínima
    # ========================================================

    if (
        not started_speaking
        or speech_ms < MIN_SPEECH_MS
    ):

        return ""

    # ========================================================
    # Resultado final Vosk
    # ========================================================

    res_final = json.loads(
        recognizer.FinalResult()
    )

    txt_final = (
        res_final
        .get("text", "")
        .strip()
    )

    if txt_final:

        texto_final_acumulado += txt_final

    return texto_final_acumulado.strip()


# ============================================================
# ================== BUCLE PRINCIPAL =========================
# ============================================================

def start_kipp_chat():

    print(
        f"--- Chat con KIPP por voz "
        f"(Vosk + Ollama + {MODEL_NAME}) ---"
    )

    print(
        "Habla una pregunta. "
        "Di 'salir' para terminar."
    )

    print()

    counter = 0

    while True:

        # ====================================================
        # ESCUCHAR
        # ====================================================

        user_question = transcribir_voz()

        print(
            f"Tú (voz): {user_question}"
        )

        # ====================================================
        # ENTRADA VACÍA
        # ====================================================

        if not user_question.strip():

            print(
                "KIPP: Entrada vacía. "
                "Intenta de nuevo."
            )

            counter += 1

            if counter == 5:

                temita = str(
                    random.randint(1, 3)
                )

                subprocess.run(
                    [
                        "ffplay",
                        "-v",
                        "0",
                        "-nodisp",
                        "-autoexit",
                        temita + ".mp3"
                    ],
                    check=True,
                )

                counter = 0

            continue

        # ====================================================
        # HOLA
        # ====================================================

        if user_question.lower() in [
            "hola kip",
            "hola equipo",
            "hola"
        ]:

            despedida = (
                "Hola! Soy Kip, listo para funcionar."
            )

            print(
                "KIPP:",
                despedida
            )

            hablar(despedida)

            esperar_habla()

            continue

        # ====================================================
        # VENDE VERDURAS
        # ====================================================

        if user_question.lower() in [
            "vende verduras",
            "vende verdura"
        ]:

            subprocess.run(
                [
                    "ffplay",
                    "-v",
                    "0",
                    "-nodisp",
                    "-autoexit",
                    "2.mp3"
                ],
                check=True,
            )

            continue

        # ====================================================
        # FRASE ICÓNICA
        # ====================================================

        if user_question.lower() in [
            "tírate una frase icónica",
            "tírate un clásico"
        ]:

            subprocess.run(
                [
                    "ffplay",
                    "-v",
                    "0",
                    "-nodisp",
                    "-autoexit",
                    "1.mp3"
                ],
                check=True,
            )

            continue

        # ====================================================
        # OYE KIP
        # ====================================================

        if user_question.lower() in [
            "oye equipo",
            "kip",
            "equipo",
            "oye kip",
            "oye y kip",
            "oye"
        ]:

            despedida = "Dime."

            print(
                "KIPP:",
                despedida
            )

            hablar(despedida)

            esperar_habla()

            continue

        # ====================================================
        # SALIR
        # ====================================================

        if user_question.lower() in [
            "salir",
            "salir.",
            "exit",
            "quit",
            "terminar sesión",
            "terminar sesión.",
        ]:

            despedida = (
                "Terminando sesión. Adiós humano."
            )

            print(
                "KIPP:",
                despedida
            )

            hablar(despedida)

            esperar_habla()

            break

        # ====================================================
        # CONSULTAR QWEN
        # ====================================================

        ask_kipp_stream(
            user_question
        )

        # ====================================================
        # Esperar que RHVoice termine
        # ====================================================

        esperar_habla()


# ============================================================
# ================== MAIN ===================================
# ============================================================

if __name__ == "__main__":

    start_kipp_chat()
