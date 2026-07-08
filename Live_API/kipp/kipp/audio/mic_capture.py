"""
Captura de micrófono en tiempo real.

`sounddevice` ejecuta su callback en un hilo propio de PortAudio, así que
nunca debe bloquearse ni tocar directamente objetos asyncio desde ahí.
Usamos `loop.call_soon_threadsafe` para inyectar cada bloque de audio en
una `asyncio.Queue` que sí vive en el event loop principal.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

import sounddevice as sd

logger = logging.getLogger(__name__)


class MicrophoneCapture:
    """Produce bloques PCM16 mono en tiempo real como un stream asíncrono."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        sample_rate: int = 16000,
        channels: int = 1,
        block_ms: int = 100,
        device: Optional[int] = None,
    ) -> None:
        self._loop = loop
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = int(sample_rate * block_ms / 1000)
        self.device = device

        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)
        self._stream: Optional[sd.RawInputStream] = None

    def _on_audio_block(self, indata, frames, time_info, status) -> None:
        if status:
            logger.debug("Estado del stream de entrada: %s", status)
        # `indata` es un buffer reutilizado por PortAudio: copiamos los bytes.
        chunk = bytes(indata)
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)
        except RuntimeError:
            # El loop ya se cerró (apagado en curso); ignoramos el bloque.
            pass

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.blocksize,
            device=self.device,
            callback=self._on_audio_block,
        )
        self._stream.start()
        logger.info(
            "Micrófono iniciado (%d Hz, %d canal(es), bloque=%dms)",
            self.sample_rate,
            self.channels,
            int(self.blocksize / self.sample_rate * 1000),
        )

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        logger.info("Micrófono detenido")

    async def chunks(self) -> AsyncIterator[bytes]:
        """Itera indefinidamente sobre los bloques de audio capturados."""
        while True:
            chunk = await self._queue.get()
            yield chunk
