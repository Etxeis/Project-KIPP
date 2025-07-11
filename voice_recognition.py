from vosk import Model, KaldiRecognizer
import pyaudio
import json

model = Model("vosk-model-small-es-0.42")  # Modelo descargado de Vosk
rec = KaldiRecognizer(model, 16000)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

while True:
    data = stream.read(4000, exception_on_overflow=False)
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        print("Tú dijiste:", result["text"])
