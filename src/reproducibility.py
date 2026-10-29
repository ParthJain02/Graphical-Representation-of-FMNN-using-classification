from __future__ import annotations

import json
import platform
import random
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def collect_software_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'numpy': getattr(np, '__version__', 'not-installed'),
    }
    try:
        import sklearn  # type: ignore
        versions['scikit-learn'] = sklearn.__version__
    except Exception:
        versions['scikit-learn'] = 'not-installed'
    try:
        import yaml
        versions['pyyaml'] = yaml.__version__
    except Exception:
        versions['pyyaml'] = 'not-installed'
    if torch is not None:
        versions['torch'] = torch.__version__
    return versions


def write_metadata(path: str, payload: dict[str, Any]) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
