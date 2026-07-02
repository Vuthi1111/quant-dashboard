"""
walk_forward.py
═══════════════════════════════════════════════════════════════════════════════
Purged Walk-Forward Cross-Validation with 3-Zone Fold Structure

Fold layout per step:
  [TRAIN (expanding or rolling)] [EMBARGO] [VALIDATION Month k+1]
  [EMBARGO] [OOS TEST Month k+2]

Both expanding window and rolling window variants are generated.
Step size: 1 calendar month
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterator, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# FOLD DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WFFold:
    fold_id:     int
    window_type: str                    # 'expanding' | 'rolling'
    train_idx:   np.ndarray             # integer positions into full DataFrame
    val_idx:     np.ndarray             # validation month (month k+1)
    test_idx:    np.ndarray             # OOS test month (month k+2) — 3rd month
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    val_start:   pd.Timestamp
    val_end:     pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp

    def summary(self) -> str:
        return (f"[Fold {self.fold_id:02d} | {self.window_type:10s}] "
                f"Train: {self.train_start.date()} → {self.train_end.date()} "
                f"({len(self.train_idx):5d} bars) | "
                f"Val: {self.val_start.date()} → {self.val_end.date()} "
                f"({len(self.val_idx):4d} bars) | "
                f"OOSTest: {self.test_start.date()} → {self.test_end.date()} "
                f"({len(self.test_idx):4d} bars)")


# ─────────────────────────────────────────────────────────────────────────────
# PURGED WALK-FORWARD SPLITTER
# ─────────────────────────────────────────────────────────────────────────────

class PurgedWalkForwardSplit:
    """
    Generate purged walk-forward folds with a 3-zone structure per fold:

        [TRAIN] → [embargo_bars gap] → [VALIDATION: month k+1]
                → [embargo_bars gap] → [OOS TEST: month k+2]

    Parameters
    ----------
    index          : Full DatetimeIndex of the dataset
    min_train_months: Minimum months required before first fold begins
    embargo_bars   : Number of bars to remove at train/val and val/test boundaries
    rolling_window_months: If set, training window is fixed (rolling). If None, expanding.
    step_months    : How many months to slide forward per fold (default: 1)
    holdout_months : Final N months reserved as final test zone (never used in WFV)
    """

    def __init__(self,
                 index: pd.DatetimeIndex,
                 min_train_months: int = 12,
                 embargo_bars: int = 40,            # ~1 trading week of 1H bars
                 rolling_window_months: int = None,  # None = expanding
                 step_months: int = 1,
                 holdout_months: int = 15):

        self.index                  = index
        self.min_train_months       = min_train_months
        self.embargo_bars           = embargo_bars
        self.rolling_window_months  = rolling_window_months
        self.step_months            = step_months
        self.holdout_months         = holdout_months

        # Compute month-boundary offsets
        self._months = self._extract_month_starts()
        self._pos    = pd.Series(np.arange(len(index)), index=index)

        # Holdout boundary — final N months are the locked TEST ZONE
        total_months = len(self._months)
        self._holdout_start_idx = total_months - self.holdout_months
        self._wfv_months = self._months[:self._holdout_start_idx]

    def _extract_month_starts(self) -> list[pd.Timestamp]:
        """Return list of first bar timestamps per calendar month."""
        df_helper = pd.DataFrame({"month": self.index.to_period("M")},
                                 index=self.index)
        return df_helper.groupby("month").apply(
            lambda g: g.index[0]).tolist()

    def _bars_in_range(self, start: pd.Timestamp,
                       end: pd.Timestamp) -> np.ndarray:
        """Return integer positions for bars in [start, end]."""
        mask = (self.index >= start) & (self.index <= end)
        return self._pos[mask].values

    def _month_end(self, month_start: pd.Timestamp) -> pd.Timestamp:
        """Return the last bar timestamp within the given calendar month."""
        m = month_start.to_period("M")
        mask = self.index.to_period("M") == m
        return self.index[mask][-1] if mask.any() else month_start

    def _purge(self, train_idx: np.ndarray,
               boundary_pos: int) -> np.ndarray:
        """Remove training bars whose index position falls within embargo zone."""
        return train_idx[train_idx < boundary_pos - self.embargo_bars]

    def generate_folds(self, window_type: str = "expanding") -> Iterator[WFFold]:
        """
        Yield WFFold objects for either 'expanding' or 'rolling' window types.

        The fold structure per step i:
          - Validation month  = wfv_months[i + offset]         (month k+1)
          - OOS Test month    = wfv_months[i + offset + 1]     (month k+2 — "the third")
          - Train end         = wfv_months[i + offset - 1]     (up to month k)
          - Train start       = fixed (expanding) OR rolling N months back
        """
        offset       = self.min_train_months
        n_months     = len(self._wfv_months)
        fold_counter = 0

        # We need at least offset + 2 months to have val + test
        for i in range(offset - 1, n_months - 2, self.step_months):
            val_month_start  = self._wfv_months[i + 1]
            test_month_start = self._wfv_months[i + 2]
            train_end_ts     = self._month_end(self._wfv_months[i])

            # Val and Test boundaries
            val_end_ts  = self._month_end(val_month_start)
            test_end_ts = self._month_end(test_month_start)

            # Handle test month possibly being in holdout
            if test_month_start >= self._months[self._holdout_start_idx]:
                break

            # Train start
            if window_type == "expanding":
                train_start_ts = self.index[0]
            else:
                # Rolling: go back rolling_window_months from train_end
                rw = self.rolling_window_months or 24
                lookback_month_idx = max(0, i + 1 - rw)
                train_start_ts = self._wfv_months[lookback_month_idx]

            # Get integer positions
            train_idx_raw = self._bars_in_range(train_start_ts, train_end_ts)
            val_idx       = self._bars_in_range(val_month_start, val_end_ts)
            test_idx      = self._bars_in_range(test_month_start, test_end_ts)

            if len(val_idx) == 0 or len(test_idx) == 0:
                continue

            # Purge train tail near embargo boundary
            val_start_pos = val_idx[0] if len(val_idx) > 0 else len(self.index)
            train_idx = self._purge(train_idx_raw, val_start_pos)

            if len(train_idx) < 100:          # Skip degenerate folds
                continue

            fold_counter += 1
            yield WFFold(
                fold_id     = fold_counter,
                window_type = window_type,
                train_idx   = train_idx,
                val_idx     = val_idx,
                test_idx    = test_idx,
                train_start = self.index[train_idx[0]],
                train_end   = self.index[train_idx[-1]],
                val_start   = val_month_start,
                val_end     = val_end_ts,
                test_start  = test_month_start,
                test_end    = test_end_ts,
            )

    def holdout_idx(self) -> np.ndarray:
        """
        Return integer positions for the FINAL locked test zone.
        This is NEVER used during WFV. Used once for terminal evaluation.
        """
        holdout_start = self._months[self._holdout_start_idx]
        return self._bars_in_range(holdout_start, self.index[-1])

    def print_fold_summary(self, window_type: str = "expanding") -> None:
        print(f"\n{'═'*100}")
        print(f" Walk-Forward Folds — {window_type.upper()} WINDOW")
        print(f"{'═'*100}")
        for fold in self.generate_folds(window_type):
            print(fold.summary())
        ho_idx = self.holdout_idx()
        print(f"\n{'─'*100}")
        print(f" LOCKED FINAL HOLDOUT: {self.index[ho_idx[0]].date()} → "
              f"{self.index[ho_idx[-1]].date()} ({len(ho_idx)} bars)")
        print(f"{'═'*100}\n")
