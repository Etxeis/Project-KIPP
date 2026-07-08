"""
Cliente asíncrono para la Gemini Live API (BidiGenerateContent) vía
WebSocket "en crudo" (sin el SDK de google-genai), tal y como pide el
proyecto.

Decisiones de diseño relevantes:

- `generationConfig.responseModalities` se fija a `["TEXT"]`: le pedimos a
  Gemini que NUNCA devuelva audio. La voz la genera RHVoice, no Gemini.
- `inputAudioTranscription` se activa para que la propia API transcriba el
  audio del usuario; no usamos Vosk/Whisper en ningún punto.
- `realtimeInputConfig.activityHandling = START_OF_ACTIVITY_INTERRUPTS`
  hace que el propio servidor de Gemini detecte cuándo el usuario empieza
  a hablar y corte su turno de generación, emitiendo
  `serverContent.interrupted = true`. Ese evento es la señal de barge-in
  que usamos para cortar la voz de KIPP (ver `core/orchestrator.py`).
- Todo el envío pasa por una cola interna (`_send_queue`) para que enviar
  audio del micrófono nunca compita de forma insegura con otros mensajes
  salientes (p. ej. respuestas a tool calls en el futuro).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, Awaitable, Callable, Optional

import websockets

logger = logging.getLogger(__name__)

_LIVE_WS_HOST = "generativelanguage.googleapis.com"
_LIVE_WS_PATH = (
    "/ws/google.ai.generativelanguage.v1beta.GenerativeService."
    "BidiGenerateContent"
)

TextCallback = Callable[[str], Optional[Awaitable[None]]]
VoidCallback = Callable[[], Optional[Awaitable[None]]]


class GeminiLiveClient:
    """Encapsula la sesión WebSocket con la Gemini Live API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        system_instruction: Optional[str] = None,
        on_text_delta: Optional[TextCallback] = None,
        on_turn_complete: Optional[VoidCallback] = None,
        on_interrupted: Optional[VoidCallback] = None,
        on_input_transcript: Optional[TextCallback] = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._system_instruction = system_instruction

        self.on_text_delta = on_text_delta
        self.on_turn_complete = on_turn_complete
        self.on_interrupted = on_interrupted
        self.on_input_transcript = on_input_transcript

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._send_queue: "asyncio.Queue[dict]" = asyncio.Queue()
        self.setup_complete = asyncio.Event()
        self._closed = asyncio.Event()

    @property
    def _url(self) -> str:
        return f"wss://{_LIVE_WS_HOST}{_LIVE_WS_PATH}?key={self._api_key}"

    async def connect(self) -> tuple[asyncio.Task, asyncio.Task]:
        """Abre la conexión, envía el `setup` y arranca las tareas de
        envío/recepción. Devuelve ambas tareas para que el llamante pueda
        esperarlas (o cancelarlas) según convenga.
        """
        self._ws = await websockets.connect(self._url, max_size=None)
        await self._send_setup()

        recv_task = asyncio.create_task(self._receive_loop(), name="gemini-recv")
        send_task = asyncio.create_task(self._send_loop(), name="gemini-send")

        await asyncio.wait_for(self.setup_complete.wait(), timeout=15)
        logger.info("Sesión Gemini Live establecida (modelo=%s)", self._model)
        return recv_task, send_task

    async def _send_setup(self) -> None:
        setup: dict[str, Any] = {
            "setup": {
                "model": f"models/{self._model}",
                # Solo texto: la síntesis de voz la hace RHVoice, no Gemini.
                "generationConfig": {"responseModalities": ["TEXT"]},
                # La propia Live API transcribe el audio de entrada.
                "inputAudioTranscription": {},
                "realtimeInputConfig": {
                    "activityHandling": "START_OF_ACTIVITY_INTERRUPTS",
                },
            }
        }
        if self._system_instruction:
            setup["setup"]["systemInstruction"] = {
                "parts": [{"text": self._system_instruction}]
            }
        await self._ws.send(json.dumps(setup))

    # ------------------------------------------------------------------ #
    # Envío
    # ------------------------------------------------------------------ #

    async def send_audio_chunk(self, pcm16_bytes: bytes) -> None:
        """Encola un bloque de audio PCM16 16kHz mono para enviarlo a
        Gemini como `realtimeInput`. No bloquea a la espera del envío
        real por WebSocket.
        """
        message = {
            "realtimeInput": {
                "audio": {
                    "data": base64.b64encode(pcm16_bytes).decode("ascii"),
                    "mimeType": "audio/pcm;rate=16000",
                }
            }
        }
        await self._send_queue.put(message)

    async def send_text(self, text: str) -> None:
        """Permite inyectar texto (útil para depuración o entrada mixta
        texto+voz) usando el mismo canal `realtimeInput`.
        """
        await self._send_queue.put({"realtimeInput": {"text": text}})

    async def _send_loop(self) -> None:
        try:
            while not self._closed.is_set():
                message = await self._send_queue.get()
                if self._ws is None:
                    break
                await self._ws.send(json.dumps(message))
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed:
            logger.info("Conexión cerrada mientras se enviaba")
        except Exception:
            logger.exception("Error en el bucle de envío hacia Gemini")

    # ------------------------------------------------------------------ #
    # Recepción
    # ------------------------------------------------------------------ #

    async def _receive_loop(self) -> None:
        try:
            async for raw_message in self._ws:
                await self._handle_server_message(json.loads(raw_message))
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed as exc:
            logger.info("Gemini cerró la conexión: %s", exc)
        except Exception:
            logger.exception("Error en el bucle de recepción de Gemini")
        finally:
            self._closed.set()

    async def _handle_server_message(self, data: dict) -> None:
        if "setupComplete" in data:
            self.setup_complete.set()
            return

        server_content = data.get("serverContent")
        if not server_content:
            # También podrían llegar `toolCall`, `goAway`, etc. Se dejan
            # como puntos de extensión para cuando se añadan tools.
            return

        if server_content.get("interrupted"):
            logger.debug("Barge-in: el usuario ha interrumpido a Gemini")
            await self._fire(self.on_interrupted)

        input_transcription = server_content.get("inputTranscription")
        if input_transcription is not None:
            text = input_transcription.get("text", "")
            if text:
                await self._fire(self.on_input_transcript, text)

        model_turn = server_content.get("modelTurn")
        if model_turn:
            for part in model_turn.get("parts", []):
                text = part.get("text")
                if text:
                    await self._fire(self.on_text_delta, text)

        if server_content.get("turnComplete"):
            await self._fire(self.on_turn_complete)

    @staticmethod
    async def _fire(callback, *args) -> None:
        if callback is None:
            return
        result = callback(*args)
        if asyncio.iscoroutine(result):
            await result

    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        self._closed.set()
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
