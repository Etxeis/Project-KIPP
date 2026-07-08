"""
Motor de texto-a-voz desacoplado del resto del sistema mediante una cola.

Objetivos que cumple este módulo (requisitos del proyecto):

1. La síntesis (RHVoice, bloqueante) y la reproducción (también
   bloqueante) NUNCA se ejecutan en el event loop de asyncio: se delegan
   a un hilo worker vía `loop.run_in_executor`, así que la comunicación
   con la Live API jamás se ve retrasada por el TTS.
2. `interrupt()` permite cortar de forma inmediata tanto el audio que se
   está reproduciendo en ese instante como cualquier frase pendiente en
   la cola: es la pieza que hace posible el barge-in.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from kipp.audio.player import AudioPlayer
from kipp.tts.rhvoice_backend import RHVoiceSynthesizer, SynthesisError

logger = logging.getLogger(__name__)

_STOP_SENTINEL = object()


class TTSEngine:
    """Cola de texto -> voz, con un único worker consumiéndola en orden.

    El worker vive en la misma tarea de asyncio (`_worker`), pero delega
    el trabajo pesado (síntesis y reproducción) a hilos separados. Todo
    el estado compartido (`_generation`) se lee/escribe únicamente desde
    el event loop, así que no hace falta un lock explícito para él.
    """

    def __init__(self, synthesizer: RHVoiceSynthesizer) -> None:
        self._synth = synthesizer
        self._player = AudioPlayer()
        self._queue: "asyncio.Queue[str | object]" = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

        # Se incrementa en cada interrupción. Cada ítem que se procesa
        # lleva "grabado" el número de generación vigente en el momento
        # de encolarse; si al llegar su turno la generación ya cambió,
        # se descarta sin sintetizar ni reproducir.
        self._generation = 0

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(
                self._worker_loop(), name="tts-worker"
            )

    async def enqueue(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        await self._queue.put((self._generation, text))

    def interrupt(self) -> None:
        """Corta la voz de KIPP inmediatamente (barge-in).

        Debe poder llamarse desde el event loop (p. ej. desde el callback
        `on_interrupted` del cliente de Gemini) de forma no bloqueante.
        """
        self._generation += 1
        self._player.stop()  # corta la reproducción en curso al instante
        self._drain_queue()
        logger.info("TTS interrumpido por barge-in")

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def stop(self) -> None:
        """Apagado ordenado: deja de aceptar trabajo y espera al worker."""
        if self._worker_task is None:
            return
        self._drain_queue()
        await self._queue.put(_STOP_SENTINEL)
        await self._worker_task
        self._worker_task = None

    async def _worker_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            item = await self._queue.get()
            if item is _STOP_SENTINEL:
                break

            generation, text = item
            if generation != self._generation:
                # Se interrumpió antes de que le tocara el turno.
                continue

            try:
                audio = await loop.run_in_executor(None, self._synth.synthesize, text)
            except SynthesisError:
                logger.exception("Fallo al sintetizar con RHVoice: %r", text)
                continue

            # Puede haberse interrumpido mientras sintetizábamos.
            if generation != self._generation:
                continue

            await loop.run_in_executor(
                None, self._player.play_blocking, audio.samples, audio.sample_rate
            )
