#!/usr/bin/env python3
"""Compatibility entry point for :mod:`trajectory_eval`.

The former evaluator silently sorted/dropped invalid poses and allowed the
same ground-truth sample to match multiple estimates.  The rigorous evaluator
now owns this command name as well as ``trajectory_eval.py``.
"""

try:  # Support both ``python tools/eval_ate.py`` and ``python -m tools.eval_ate``.
    from .trajectory_eval import main
except ImportError:  # pragma: no cover - exercised by the direct script invocation.
    from trajectory_eval import main


if __name__ == "__main__":
    raise SystemExit(main())
