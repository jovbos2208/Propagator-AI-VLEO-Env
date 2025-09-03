from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional
from dataclasses import dataclass
from tqdm import tqdm


class EarlyStopOnPlateau(BaseCallback):
    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: int = 0):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def _on_step(self) -> bool:
        # Placeholder: do nothing (requires eval pipeline to compute mean reward)
        return True


@dataclass
class ProgressBarCallback(BaseCallback):
    """Simple tqdm progress bar for SB3 training.

    If total is None, tries to read model._total_timesteps set by learn().
    """

    total: Optional[int] = None
    desc: str = "training"
    disable: bool = False

    def __post_init__(self):
        super().__init__()
        self._bar: Optional[tqdm] = None
        self._last: int = 0

    def _on_training_start(self) -> None:
        total = self.total
        if total is None:
            total = getattr(self.model, "_total_timesteps", None)
        try:
            total = int(total) if total is not None else None
        except Exception:
            total = None
        self._bar = tqdm(total=total, desc=self.desc, disable=self.disable)
        self._last = 0
        # Touch TensorBoard logger so event files are created immediately
        try:
            self.logger.record("progress/initialized", 1)
            # step 0 ensures TB event file exists from the start
            self.logger.dump(step=0)
        except Exception:
            pass

    def _on_step(self) -> bool:
        if self._bar is not None:
            cur = int(self.num_timesteps)
            delta = max(0, cur - self._last)
            if delta:
                self._bar.update(delta)
                self._last = cur
        return True

    def _on_training_end(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
