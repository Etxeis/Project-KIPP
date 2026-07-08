"""
`Kipp` es la clase que cablea todos los módulos entre sí:

    Micrófono --(PCM16)--> GeminiLiveClient --(texto en streaming)-->
        SentenceAccumulator --(frases)--> TTSEngine --(voz)--> Altavoz

Y en sentido contrario, la señal de barge-in de Gemini
(`serverContent.interrupted`) corta el TTSEngine al instante.

Este módulo no sabe nada de detalles de bajo nivel (formato de mensajes
WebSocket, cómo se invoca RHVoice, etc.): solo orquesta. Así, añadir
capacidades nuevas (function calling, memoria, un wake word, un segundo
canal de salida...) implica tocar este archivo y poco más.
"""

from __future__ import annotations

import asyncio
import logging

from kipp.audio.mic_capture import MicrophoneCapture
from kipp.config import KippConfig
from kipp.live.client import GeminiLiveClient
from kipp.text.sentence_splitter import SentenceAccumulator
from kipp.tts.rhvoice_backend import RHVoiceSynthesizer
from kipp.tts.speech_queue import TTSEngine

logger = logging.getLogger(__name__)


class Kipp:
    def __init__(self, config: KippConfig) -> None:
        self.config = config

        self._sentences = SentenceAccumulator()
        self._tts = TTSEngine(
            synthesizer=RHVoiceSynthesizer(
                voice=config.rhvoice_voice,
                rate=config.rhvoice_rate,
                binary=config.rhvoice_binary,
            )
        )
        self._client = GeminiLiveClient(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
            system_instruction=config.system_instruction,
            on_text_delta=self._on_text_delta,
            on_turn_complete=self._on_turn_complete,
            on_interrupted=self._on_interrupted,
            on_input_transcript=self._on_input_transcript,
        )
        self._mic: MicrophoneCapture | None = None

    # ------------------------------------------------------------------ #
    # Callbacks del cliente Gemini
    # ------------------------------------------------------------------ #

    async def _on_text_delta(self, delta: str) -> None:
        for sentence in self._sentences.feed(delta):
            logger.debug("KIPP dice: %s", sentence)
            await self._tts.enqueue(sentence)

    async def _on_turn_complete(self) -> None:
        for sentence in self._sentences.flush():
            logger.debug("KIPP dice (cierre de turno): %s", sentence)
            await self._tts.enqueue(sentence)

    async def _on_interrupted(self) -> None:
        # Barge-in: el usuario ha empezado a hablar mientras KIPP
        # respondía. Cortamos su voz y descartamos lo que quedaba por
        # decir, tanto en el acumulador de frases como en la cola de TTS.
        self._sentences.reset()
        self._tts.interrupt()

    async def _on_input_transcript(self, text: str) -> None:
        logger.info("Usuario: %s", text)

    # ------------------------------------------------------------------ #
    # Ciclo de vida
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._mic = MicrophoneCapture(
            loop,
            sample_rate=self.config.input_sample_rate,
            channels=self.config.input_channels,
            block_ms=self.config.input_block_ms,
        )

        await self._tts.start()
        recv_task, send_task = await self._client.connect()
        self._mic.start()
        mic_task = asyncio.create_task(self._pump_microphone(), name="mic-pump")

        logger.info("KIPP está escuchando. Ctrl+C para salir.")
        try:
            done, pending = await asyncio.wait(
                {recv_task, send_task, mic_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                exc = task.exception()
                if exc:
                    raise exc
        finally:
            await self._shutdown(recv_task, send_task, mic_task)

    async def _pump_microphone(self) -> None:
        assert self._mic is not None
        async for chunk in self._mic.chunks():
            await self._client.send_audio_chunk(chunk)

    async def _shutdown(self, *tasks: asyncio.Task) -> None:
        logger.info("Cerrando KIPP...")
        if self._mic is not None:
            self._mic.stop()

        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        await self._tts.stop()
        await self._client.close()
        logger.info("KIPP se ha detenido")
