"""
Backend de síntesis de voz basado en RHVoice.

Se apoya en el binario de línea de comandos `RHVoice-test`, que viene con
cualquier instalación estándar de RHVoice (paquete `rhvoice` /
`RHVoice-utils` según la distro). Se invoca como proceso externo y se le
pasa el texto por stdin, generando un WAV temporal que luego se carga
como array numpy PCM16.

Esta clase es intencionadamente "tonta" y síncrona/bloqueante: toda la
gestión de concurrencia (no bloquear el event loop, permitir cancelación)
vive en `tts/speech_queue.py`. Mantenerlo así facilita sustituir RHVoice
por otro motor en el futuro sin tocar el resto del sistema.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


class SynthesisError(RuntimeError):
    """Fallo al invocar RHVoice o al leer el WAV resultante."""


@dataclass(frozen=True)
class SynthesizedAudio:
    samples: np.ndarray  # PCM16 mono
    sample_rate: int


class RHVoiceSynthesizer:
    def __init__(
        self,
        voice: str = "Anna",
        rate: int = 50,
        binary: str = "RHVoice-test",
        timeout_s: float = 20.0,
    ) -> None:
        self.voice = voice
        self.rate = rate
        self.binary = binary
        self.timeout_s = timeout_s

    def synthesize(self, text: str) -> SynthesizedAudio:
        """Llamada BLOQUEANTE. Debe ejecutarse siempre en un hilo worker
        (ver `TTSEngine`), nunca directamente en el event loop de asyncio.
        """
        if not text.strip():
            raise SynthesisError("Texto vacío recibido para síntesis")

        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            cmd = [
                self.binary,
                "-p", self.voice,
                "-r", str(self.rate),
                "-o", out_path,
            ]
            result = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=self.timeout_s,
            )
            if result.returncode != 0:
                raise SynthesisError(
                    f"RHVoice-test devolvió código {result.returncode}: "
                    f"{result.stderr.decode(errors='replace')}"
                )
            return self._load_wav(out_path)
        except FileNotFoundError as exc:
            raise SynthesisError(
                f"No se encontró el binario '{self.binary}'. ¿Está RHVoice "
                "instalado y en el PATH?"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SynthesisError("RHVoice tardó demasiado en sintetizar") from exc
        finally:
            try:
                os.remove(out_path)
            except OSError:
                pass

    @staticmethod
    def _load_wav(path: str) -> SynthesizedAudio:
        with wave.open(path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16)
        return SynthesizedAudio(samples=samples, sample_rate=sample_rate)
