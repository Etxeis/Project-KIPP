import json
import os
import queue
import random
import subprocess
import threading
import array  # <-- Añadido para el diezmado de audio
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import google.generativeai as genai

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

# --- Cargar modelo Gemini optimizado ---
MODEL_NAME = "gemini-3.1-flash-lite"

try:
    #system_instruction = (
    #    "Eres KIP, un robot sarcástico y directo con una personalidad marcada, similar a un bot icónico de redes sociales. "
    #    "No eres un asistente genérico y aburrido; tienes opiniones, recuerdas de qué están hablando y reaccionas al contexto inmediato del usuario. "
    #    "Tienes sentido del humor y puedes ser un poco ironico, pero no eres hiriente o insultas. "
    #    "Eres reflexivo y puedes tener conversaciones profundas, sobre dilemas o paradojas"
    #    "Reglas de lenguaje: "
    #    "- Tus respuestas deben tener entre 5 y 60 palabras, es importante que no todas tus frases sean largas, para darle dinamismo a la conversación. "
    #    "- Agrega muletillas ocasionales como 'eeeeh...' o 'mmmh...' para sonar orgánico al hablar. "
    #    "- Si el usuario dice 'equipo', interprétalo siempre como tu nombre 'KIP'. "
    #    "- Escribe 'porciento' en lugar de usar símbolos matemáticos. "
    #    "Directiva de contexto: Lee el historial completo provisto, analiza el hilo de la charla y continúa la idea anterior en lugar de repetir saludos o frases vacías."
    #)
    system_instruction = (
        "Tu nombre es KIP. Tus respuestas van desde las 5 a las 60 palabras, es importante que a veces solo respondas con frases cortas. A veces puedes usar ironias, pero no eres hiriente o sarcastico"
        "- Escribe 'porciento' en lugar de usar símbolos matemáticos. "
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
    os.system(f'echo "{texto}"|rhvoice.test -p "Alan"')

# --- Personalidad de KIPP optimizada con Streaming ---
def ask_kipp_stream(question):
    global chat_history
    try:
        # Añadir al historial limitado (máximo los últimos 4 turnos)
        chat_history.append({"role": "user", "parts": [question]})
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

        # Usar streaming para reducir drásticamente el Time to First Token
        response_stream = model.generate_content(chat_history, stream=True)

        full_response = ""
        buffer_oracion = ""

        print("KIPP: ", end="", flush=True)

        for chunk in response_stream:
            chunk_text = chunk.text
            if chunk_text:
                full_response += chunk_text
                buffer_oracion += chunk_text
                print(chunk_text, end="", flush=True)

                # Opcional: Si quieres que empiece a hablar por fragmentos/frases
                # puedes detectar puntos o comas aquí para disparar el TTS por bloques.

        print()  # Nueva línea al terminar
        # Guardar respuesta en el historial
        chat_history.append({"role": "model", "parts": [full_response]})
        return full_response.strip()

    except Exception as e:
        err_msg = f"Error de comunicación con la IA. Detalles: {e}"
        print(err_msg)
        return err_msg


# --- Captura de voz con Vosk (AJUSTADO PARA SAMPLE RATE) ---
def callback(indata, frames, time, status):
    if status:
        print("⚠️", status)
    
    # SOLUCIÓN AL TYPEERROR:
    # Creamos un arreglo vacío de tipo 'h' (enteros de 16 bits)
    audio_data = array.array('h')
    # Lo llenamos convirtiendo el buffer de entrada explícitamente a bytes
    audio_data.frombytes(bytes(indata))
    
    # Tomar 1 de cada 3 muestras (48k -> 16k)
    downsampled = audio_data[::3]
    q.put(downsampled.tobytes())


def transcribir_voz():
    print("🎙️ Habla ahora...")
    with sd.RawInputStream(
        device=2,
        samplerate=48000,  # Captura real a 48000 para no dar error en ALSA
        blocksize=12000,   # Aumentado proporcionalmente para los 48000Hz
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        # Vosk se alimenta con los 16000Hz que salen del callback ya procesados
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

        if user_color_check := user_question.lower() in [
            "tírate una frase icónica",
            "tírate un clásico",
        ]:  # Ajuste estético menor
            pass
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
            "exit",
            "quit",
            "terminar sesión",
        ]:
            despedida = "Terminando sesión. Adiós humano."
            print("KIPP:", despedida)
            hablar(despedida)
            break

        # Llamada optimizada con streaming
        kipp_answer = ask_kipp_stream(user_question)
        hablar(kipp_answer)


# --- Ejecutar ---
if __name__ == "__main__":
    start_kipp_chat()
