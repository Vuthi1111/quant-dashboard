"""
model_stack.py
═══════════════════════════════════════════════════════════════════════════════
Full Model Stack for Adaptive Walk-Forward Pipeline — NAS100

Layers:
  L1 — Logistic Regression     (linear baseline, interpretable)
  L1 — LightGBM                (gradient boosted trees, primary signal)
  L1 — LSTM (PyTorch)          (sequence model over last N bars)
  L2 — Meta-Learner            (stacking: LR over L1 OOF predictions)

All preprocessing (scaler, PCA, imputer) is fit on TRAIN only inside each fold.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING PIPELINE (in-fold, no leakage)
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessor(n_components: int = 20) -> Pipeline:
    """
    Returns an sklearn Pipeline: impute → scale → PCA.
    Must be fit only on training data, then transform val/test.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("pca",     PCA(n_components=n_components, whiten=True)),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# L1 MODEL 1 — LOGISTIC REGRESSION (BASELINE)
# ─────────────────────────────────────────────────────────────────────────────

def train_logistic(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray,   y_val: np.ndarray) -> LogisticRegression:
    """Calibrated logistic regression baseline."""
    lr = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs",
                            class_weight="balanced")
    cal = CalibratedClassifierCV(lr, cv=3, method="sigmoid")
    cal.fit(X_train, y_train)
    return cal


# ─────────────────────────────────────────────────────────────────────────────
# L1 MODEL 2 — LIGHTGBM WITH OPTUNA HPO
# ─────────────────────────────────────────────────────────────────────────────

def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray,   y_val: np.ndarray,
                   n_trials: int = 30) -> lgb.Booster:
    """
    LightGBM classifier with Bayesian hyperparameter optimisation on validation fold.
    Only validation AUC is used to select hyperparameters — test data is never seen.

    Fix (LightGBM 4.x): Dataset is recreated fresh inside each Optuna trial and
    feature_pre_filter=False is set so min_data_in_leaf can vary freely without
    triggering the cached feature-filter conflict.
    """
    def objective(trial):
        params = {
            "objective":          "binary",
            "metric":             "auc",
            "verbosity":          -1,
            "boosting_type":      "gbdt",
            "feature_pre_filter": False,        # Fix: allow min_data_in_leaf to change
            "learning_rate":      trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves":         trial.suggest_int("num_leaves", 20, 150),
            "max_depth":          trial.suggest_int("max_depth", 3, 10),
            "min_child_samples":  trial.suggest_int("min_child_samples", 20, 100),
            "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        n_est = trial.suggest_int("n_estimators", 100, 600)
        # Recreate Dataset each trial — prevents feature_pre_filter cache conflict
        dtrain_t = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        dval_t   = lgb.Dataset(X_val,   label=y_val,   free_raw_data=False,
                               reference=dtrain_t)
        model = lgb.train(params, dtrain_t, num_boost_round=n_est,
                          valid_sets=[dval_t],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(-1)])
        pred = model.predict(X_val)
        return roc_auc_score(y_val, pred)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Final refit with best hyperparameters
    best = study.best_params.copy()
    n_est = best.pop("n_estimators")
    best.update({"objective": "binary", "metric": "auc", "verbosity": -1,
                 "feature_pre_filter": False})
    dtrain_f = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dval_f   = lgb.Dataset(X_val,   label=y_val,   free_raw_data=False,
                           reference=dtrain_f)
    model = lgb.train(best, dtrain_f, num_boost_round=n_est,
                      valid_sets=[dval_f],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# L1 MODEL 3 — LSTM (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """Bidirectional LSTM → Dense → Sigmoid for binary classification."""

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=True)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])   # take last timestep
        return self.head(out).squeeze(-1)


def make_sequences(X: np.ndarray, seq_len: int = 20) -> np.ndarray:
    """Reshape flat feature matrix into 3D sequences (samples, seq_len, features)."""
    seqs = []
    for i in range(seq_len, len(X) + 1):
        seqs.append(X[i - seq_len:i])
    return np.array(seqs)


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_val:   np.ndarray, y_val: np.ndarray,
               seq_len: int = 20,
               epochs: int = 30,
               batch_size: int = 256,
               lr: float = 1e-3) -> tuple:
    """
    Train LSTM on sequential windows of the feature matrix.
    Returns (trained model, seq_len) tuple.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build sequences
    X_tr_seq = make_sequences(X_train, seq_len)
    y_tr_seq = y_train[seq_len - 1:]
    X_va_seq = make_sequences(X_val, seq_len)
    y_va_seq = y_val[seq_len - 1:]

    if len(X_tr_seq) < batch_size or len(X_va_seq) == 0:
        return None, seq_len   # Not enough data for this fold

    tr_ds = TensorDataset(torch.tensor(X_tr_seq, dtype=torch.float32),
                          torch.tensor(y_tr_seq, dtype=torch.float32))
    va_ds = TensorDataset(torch.tensor(X_va_seq, dtype=torch.float32),
                          torch.tensor(y_va_seq, dtype=torch.float32))

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=False)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    model = LSTMClassifier(input_size=X_train.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit  = nn.BCELoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_loss = np.inf
    best_state    = None
    patience_ctr  = 0
    patience      = 8

    for epoch in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # Validation loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(crit(pred, yb).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model.to("cpu"), seq_len


def predict_lstm(model, seq_len: int, X: np.ndarray) -> np.ndarray:
    """Return per-bar probability predictions. First seq_len-1 bars get nan."""
    if model is None or len(X) < seq_len:
        return np.full(len(X), np.nan)
    seqs   = make_sequences(X, seq_len)
    tensor = torch.tensor(seqs, dtype=torch.float32)
    with torch.no_grad():
        probs = model(tensor).numpy()
    full = np.full(len(X), np.nan)
    full[seq_len - 1:] = probs
    return full


# ─────────────────────────────────────────────────────────────────────────────
# L2 — META LEARNER (STACKING)
# ─────────────────────────────────────────────────────────────────────────────

class MetaLearner:
    """
    Logistic Regression meta-learner that blends L1 model predictions.
    Trained on validation-fold OOF predictions to avoid leakage.
    """
    def __init__(self):
        self.model = LogisticRegression(C=1.0, max_iter=500)

    def fit(self, p_lr: np.ndarray, p_lgbm: np.ndarray,
            p_lstm: np.ndarray, y: np.ndarray):
        """p_* are probabilities from each L1 model on validation fold."""
        X_meta = self._stack(p_lr, p_lgbm, p_lstm)
        mask   = ~np.isnan(X_meta).any(axis=1)
        self.model.fit(X_meta[mask], y[mask])

    def predict_proba(self, p_lr: np.ndarray, p_lgbm: np.ndarray,
                      p_lstm: np.ndarray) -> np.ndarray:
        X_meta = self._stack(p_lr, p_lgbm, p_lstm)
        mask   = ~np.isnan(X_meta).any(axis=1)
        out    = np.full(len(X_meta), np.nan)
        if mask.sum() > 0:
            out[mask] = self.model.predict_proba(X_meta[mask])[:, 1]
        return out

    @staticmethod
    def _stack(*arrs) -> np.ndarray:
        return np.column_stack(arrs)


# ─────────────────────────────────────────────────────────────────────────────
# FOLD RUNNER — ties everything together per fold
# ─────────────────────────────────────────────────────────────────────────────

class FoldResult:
    """Container for predictions and metrics from a single WFV fold."""
    def __init__(self, fold_id: int, window_type: str):
        self.fold_id     = fold_id
        self.window_type = window_type
        self.val_preds   = {}
        self.test_preds  = {}
        self.val_labels  = None
        self.test_labels = None
        self.metrics     = {}
        self.feature_importance: pd.Series | None = None


def run_fold(fold,
             X_full: np.ndarray,
             y_full: np.ndarray,
             feature_names: list,
             n_pca_components: int = 25,
             lgbm_trials: int = 25,
             lstm_seq_len: int = 20,
             lstm_epochs: int = 25,
             news_mask_full: np.ndarray = None) -> FoldResult:
    """
    Execute the full model stack for a single walk-forward fold.

    Preprocessing is fit only on train data.
    L1 models trained on train, evaluated on val (for HPO).
    Meta-learner trained on val predictions.
    Final ensemble evaluated on OOS test fold (the "third month").
    """
    result = FoldResult(fold.fold_id, fold.window_type)

    # ── Extract splits
    X_tr = X_full[fold.train_idx]
    y_tr = y_full[fold.train_idx]
    X_va = X_full[fold.val_idx]
    y_va = y_full[fold.val_idx]
    X_te = X_full[fold.test_idx]
    y_te = y_full[fold.test_idx]

    # Apply news mask — zero out news-adjacent rows (soft suppression)
    if news_mask_full is not None:
        nm_tr = news_mask_full[fold.train_idx].astype(bool)
        nm_va = news_mask_full[fold.val_idx].astype(bool)
        X_tr[nm_tr] = 0
        X_va[nm_va] = 0

    result.val_labels  = y_va
    result.test_labels = y_te

    # ── Preprocessing (fit on TRAIN only)
    prep = build_preprocessor(n_components=min(n_pca_components, X_tr.shape[1] - 1))
    X_tr_p = prep.fit_transform(X_tr)
    X_va_p = prep.transform(X_va)
    X_te_p = prep.transform(X_te)

    # ── L1 Model 1: Logistic Regression
    print(f"  [Fold {fold.fold_id}|{fold.window_type}] Training LogReg...")
    lr_model = train_logistic(X_tr_p, y_tr, X_va_p, y_va)
    p_lr_va  = lr_model.predict_proba(X_va_p)[:, 1]
    p_lr_te  = lr_model.predict_proba(X_te_p)[:, 1]

    # ── L1 Model 2: LightGBM
    print(f"  [Fold {fold.fold_id}|{fold.window_type}] Training LightGBM ({lgbm_trials} trials)...")
    lgbm_model = train_lightgbm(X_tr_p, y_tr, X_va_p, y_va, n_trials=lgbm_trials)
    p_lgbm_va  = lgbm_model.predict(X_va_p)
    p_lgbm_te  = lgbm_model.predict(X_te_p)

    # Feature importance (LightGBM — PCA components)
    result.feature_importance = pd.Series(lgbm_model.feature_importance("gain"),
                                          name=f"fold_{fold.fold_id}")

    # ── L1 Model 3: LSTM
    print(f"  [Fold {fold.fold_id}|{fold.window_type}] Training LSTM...")
    lstm_model, seq_len = train_lstm(X_tr_p, y_tr, X_va_p, y_va,
                                     seq_len=lstm_seq_len, epochs=lstm_epochs)
    p_lstm_va = predict_lstm(lstm_model, seq_len, X_va_p)
    p_lstm_te = predict_lstm(lstm_model, seq_len, X_te_p)

    # ── L2 Meta-Learner (trained on VALIDATION predictions)
    print(f"  [Fold {fold.fold_id}|{fold.window_type}] Fitting meta-learner...")
    # For meta training, only use rows where all L1 preds are valid
    meta = MetaLearner()
    meta.fit(p_lr_va, p_lgbm_va, p_lstm_va, y_va)

    p_meta_va = meta.predict_proba(p_lr_va, p_lgbm_va, p_lstm_va)
    p_meta_te = meta.predict_proba(p_lr_te, p_lgbm_te, p_lstm_te)

    # ── Store predictions
    result.val_preds  = {"LR": p_lr_va,  "LGBM": p_lgbm_va,
                         "LSTM": p_lstm_va, "Meta": p_meta_va}
    result.test_preds = {"LR": p_lr_te,  "LGBM": p_lgbm_te,
                         "LSTM": p_lstm_te, "Meta": p_meta_te}

    # ── Compute metrics for OOS Test (the third month)
    for name, pred in result.test_preds.items():
        valid = ~np.isnan(pred)
        if valid.sum() < 10:
            continue
        try:
            result.metrics[name] = {
                "auc":      roc_auc_score(y_te[valid], pred[valid]),
                "accuracy": accuracy_score(y_te[valid], (pred[valid] > 0.5).astype(int)),
                "logloss":  log_loss(y_te[valid], pred[valid]),
            }
        except Exception:
            pass

    print(f"  ✓ Fold {fold.fold_id} | {fold.window_type} | "
          f"OOS Meta AUC: {result.metrics.get('Meta', {}).get('auc', 0):.4f}")
    return result
