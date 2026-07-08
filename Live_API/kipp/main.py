"""Punto de entrada de KIPP.

Uso:
    export GEMINI_API_KEY="tu-clave"
    python main.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

from kipp.config import KippConfig
from kipp.core.orchestrator import Kipp


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> int:
    config = KippConfig()
    _setup_logging(config.log_level)
    logger = logging.getLogger("kipp.main")

    try:
        config.validate()
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    kipp = Kipp(config)
    try:
        asyncio.run(kipp.run())
    except KeyboardInterrupt:
        logger.info("Interrumpido por el usuario")
    return 0


if __name__ == "__main__":
    sys.exit(main())
