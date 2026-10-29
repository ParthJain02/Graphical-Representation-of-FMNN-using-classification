from __future__ import annotations

from pathlib import Path
from typing import Any


def write_summary(path: str, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as fh:
        fh.write('Experiment summary\n')
        for key, value in payload.items():
            fh.write(f'{key}: {value}\n')
