"""Test configuration.

The modules under ``src/`` import each other as top-level modules
(``from distributions import ...``), which is how the training scripts run them
(they put ``src/`` on ``sys.path``). The tests do the same.

Only ``torch`` and ``pytest`` are required: none of the tests import Lightning,
Hydra, wandb, lpips, TorchRL or DMControl.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
