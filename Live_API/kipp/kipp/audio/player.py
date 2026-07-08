"""
Reproducción de audio sintetizado, con soporte de interrupción inmediata.

Este módulo es intencionadamente independiente de RHVoice: recibe un array
numpy PCM16 y una frecuencia de muestreo, sin saber nada de cómo se generó
el audio. Esto permite sustituir el motor TTS en el futuro sin tocar esta
pieza.
"""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Reproduce PCM16 de forma bloqueante (pensado para correr en un hilo)
    pero interrumpible desde cualquier otro hilo mediante `stop()`.
    """

    def play_blocking(self, samples: np.ndarray, sample_rate: int) -> None:
        """Bloquea hasta que termina la reproducción o se llama a `stop()`.

        Debe invocarse desde un hilo worker (p. ej. vía
        `loop.run_in_executor`), nunca directamente desde el event loop.
        """
        try:
            sd.play(samples, sample_rate)
            sd.wait()
        except Exception:
            logger.exception("Error durante la reproducción de audio")

    def stop(self) -> None:
        """Corta la reproducción en curso de forma inmediata.

        Es seguro llamarlo desde cualquier hilo (incluido el event loop),
        y es la pieza clave para el barge-in: cuando Gemini nos avisa de
        que el usuario ha empezado a hablar, cortamos la voz de KIPP al
        instante.
        """
        sd.stop()
