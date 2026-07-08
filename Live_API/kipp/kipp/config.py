"""
Configuración central de KIPP.

Toda la configuración se resuelve a partir de variables de entorno, con
valores por defecto razonables. Mantener la configuración en un único
lugar facilita el crecimiento del proyecto (nuevos módulos solo importan
`KippConfig`, no leen `os.environ` directamente).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


@dataclass(frozen=True)
class KippConfig:
    # --- Gemini Live API ---
    gemini_api_key: str = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", "")
    )
    gemini_model: str = field(
        default_factory=lambda: os.environ.get(
            "GEMINI_MODEL", "gemini-3.1-flash-live-preview"
        )
    )
    system_instruction: str = field(
        default_factory=lambda: os.environ.get(
            "KIPP_SYSTEM_INSTRUCTION",
            "Eres KIPP, un robot asistente. Responde de forma breve, clara y "
            "natural, como en una conversación hablada. Responde en español "
            "salvo que el usuario te hable en otro idioma.",
        )
    )

    # --- Audio de entrada (micrófono -> Gemini) ---
    input_sample_rate: int = 16000
    input_channels: int = 1
    input_block_ms: int = 100  # tamaño de bloque enviado al websocket

    # --- RHVoice (texto -> voz) ---
    rhvoice_binary: str = field(
        default_factory=lambda: os.environ.get("RHVOICE_BINARY", "RHVoice-test")
    )
    rhvoice_voice: str = field(
        default_factory=lambda: os.environ.get("RHVOICE_VOICE", "Anna")
    )
    rhvoice_rate: int = field(
        default_factory=lambda: _env_int("RHVOICE_RATE", 50)
    )

    # --- Logging ---
    log_level: str = field(
        default_factory=lambda: os.environ.get("KIPP_LOG_LEVEL", "INFO")
    )

    def validate(self) -> None:
        if not self.gemini_api_key:
            raise ValueError(
                "Falta GEMINI_API_KEY. Exporta la variable de entorno con tu "
                "clave de la API de Gemini antes de ejecutar KIPP."
            )
