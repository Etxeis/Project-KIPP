# KIPP

Robot conversacional por voz en Python, construido sobre la **Gemini Live
API** (WebSocket) y **RHVoice**.

## Arquitectura

```
Micrófono ──PCM16──▶ GeminiLiveClient ──texto (streaming)──▶ SentenceAccumulator
                          │                                        │
                          │ serverContent.interrupted              │ frases completas
                          ▼ (barge-in)                              ▼
                      TTSEngine.interrupt()  ◀───────────────  TTSEngine.enqueue()
                                                                     │
                                                                     ▼
                                                              RHVoice + altavoz
```

```
kipp/
├── config.py                  # configuración centralizada (env vars)
├── audio/
│   ├── mic_capture.py          # captura de micrófono → asyncio.Queue
│   └── player.py                # reproducción de PCM16, interrumpible
├── live/
│   └── client.py                # WebSocket con la Gemini Live API
├── text/
│   └── sentence_splitter.py     # trocea el streaming de texto en frases
├── tts/
│   ├── rhvoice_backend.py       # invoca RHVoice-test, devuelve PCM16
│   └── speech_queue.py          # cola async + hilo worker + barge-in
└── core/
    └── orchestrator.py          # cablea todo lo anterior (clase `Kipp`)
main.py                          # punto de entrada
```

## Decisiones clave (según los requisitos)

- **Transcripción**: la hace la propia Gemini Live API
  (`inputAudioTranscription` en el mensaje `setup`). No se usa Vosk ni
  Whisper en ningún punto del código.
- **Salida de Gemini**: `generationConfig.responseModalities = ["TEXT"]`.
  Gemini nunca genera audio; solo texto, que se convierte a voz con
  RHVoice.
- **TTS no bloqueante**: `TTSEngine` (en `tts/speech_queue.py`) mantiene
  una `asyncio.Queue` consumida por un único worker; la síntesis
  (subprocess de RHVoice) y la reproducción (`sounddevice`) se ejecutan
  siempre vía `loop.run_in_executor`, por lo que jamás bloquean el bucle
  de eventos que habla con la Live API.
- **Barge-in**: se activa `realtimeInputConfig.activityHandling =
  START_OF_ACTIVITY_INTERRUPTS` en el `setup`, de modo que es el propio
  servidor de Gemini quien detecta cuándo el usuario empieza a hablar y
  envía `serverContent.interrupted = true`. Al recibirlo, `Kipp`
  descarta el buffer de frases pendientes y llama a
  `TTSEngine.interrupt()`, que corta la reproducción en curso
  (`sounddevice.stop()`) y vacía la cola.
- **Asyncio de punta a punta**: la conexión WebSocket, el envío de audio
  y la recepción de eventos son 100% asíncronos (`websockets` +
  `asyncio`). Los únicos puntos bloqueantes (RHVoice, reproducción,
  captura de audio) están explícitamente aislados en hilos.

## Requisitos del sistema

- Python 3.10+
- [RHVoice](https://github.com/RHVoice/RHVoice) instalado, con el binario
  `RHVoice-test` en el `PATH` y al menos una voz instalada (p. ej.
  `Anna` para español/ruso, `Alan`, etc. — usa `RHVoice-test -L` para
  listar las voces disponibles en tu sistema).
- Un dispositivo de entrada (micrófono) y uno de salida (altavoces)
  accesibles vía PortAudio.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuración

Variables de entorno (todas opcionales salvo `GEMINI_API_KEY`):

| Variable                | Descripción                                   | Por defecto                        |
|--------------------------|------------------------------------------------|-------------------------------------|
| `GEMINI_API_KEY`          | Clave de la API de Gemini                       | *(obligatoria)*                     |
| `GEMINI_MODEL`             | Modelo Live a usar                              | `gemini-3.1-flash-live-preview`     |
| `KIPP_SYSTEM_INSTRUCTION`  | Instrucción de sistema para Gemini              | Persona por defecto de KIPP         |
| `RHVOICE_BINARY`           | Ruta/nombre del binario de RHVoice              | `RHVoice-test`                      |
| `RHVOICE_VOICE`            | Voz de RHVoice a usar                           | `Anna`                              |
| `RHVOICE_RATE`             | Velocidad de habla (0-100)                       | `50`                                |
| `KIPP_LOG_LEVEL`           | Nivel de logging                                | `INFO`                              |

## Ejecución

```bash
export GEMINI_API_KEY="tu-clave"
python main.py
```

Habla con normalidad; KIPP transcribe, piensa y responde por voz. Si
empiezas a hablar mientras KIPP está respondiendo, su voz se corta al
instante y Gemini atiende tu nueva intervención (barge-in).

## Puntos de extensión

- **Function calling / tools**: `GeminiLiveClient` ya deja sitio para
  gestionar `toolCall` / `toolCallCancellation` en
  `_handle_server_message`; basta con añadir los callbacks
  correspondientes y las definiciones de `tools` en el `setup`.
- **Wake word / activación por palabra clave**: se integraría como un
  módulo nuevo bajo `audio/`, que decida cuándo llamar a
  `mic.start()` / controlar el flujo hacia `GeminiLiveClient`.
- **Otro motor de TTS**: basta con implementar una clase con el mismo
  contrato que `RHVoiceSynthesizer.synthesize(text) -> SynthesizedAudio`
  e inyectarla en `TTSEngine`.
- **Memoria / contexto persistente**: puede añadirse como un módulo bajo
  `core/`, alimentando `system_instruction` o usando
  `BidiGenerateContentClientContent` para inyectar historial.
