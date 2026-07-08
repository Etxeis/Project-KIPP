"""
Gemini entrega el texto de su respuesta a trozos pequeños ("deltas"), no
frase a frase. Si esperáramos al `turnComplete` para empezar a sintetizar
voz, KIPP tardaría en responder. Si sintetizáramos cada delta por separado,
la prosodia sería pésima (RHVoice recibiría fragmentos de palabras).

`SentenceAccumulator` resuelve esto: va acumulando el texto entrante y
libera frases completas (delimitadas por puntuación de cierre) en cuanto
están disponibles, manteniendo el resto en buffer.
"""

from __future__ import annotations

import re
from typing import List

_SENTENCE_BOUNDARY = re.compile(r"(?<=[\.\!\?\:\;…])\s+")


class SentenceAccumulator:
    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, delta: str) -> List[str]:
        """Añade un fragmento de texto y devuelve las frases completas
        que ya se pueden enviar a síntesis."""
        self._buffer += delta
        parts = _SENTENCE_BOUNDARY.split(self._buffer)

        if len(parts) <= 1:
            return []

        *complete_sentences, remainder = parts
        self._buffer = remainder
        return [s.strip() for s in complete_sentences if s.strip()]

    def flush(self) -> List[str]:
        """Libera cualquier resto pendiente (llamar al recibir
        `turnComplete` o al detectar una interrupción)."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return [remaining] if remaining else []

    def reset(self) -> None:
        """Descarta el buffer sin devolverlo (usar en barge-in: lo que
        Gemini iba a decir ya no importa)."""
        self._buffer = ""
