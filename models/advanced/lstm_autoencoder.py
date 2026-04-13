"""
models/advanced/lstm_autoencoder.py
-------------------------------------
Advanced anomaly detector using an Autoencoder.

PyTorch path  : LSTM Autoencoder (when torch available)
Fallback path : sklearn MLPRegressor as encoder + decoder (stable, tested)

Both share the same API: fit / predict / save / load.

Anomaly score = reconstruction error (MSE per user).
Top contributing features = features with highest per-feature reconstruction error.
"""

import os, json, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor

try:
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except (ImportError, TypeError):
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

# Must be defined before any class — pickle resolves this at module load time
MODEL_DIR = "models/advanced/saved"


# ════════════════════════════════════════════════════════════════
# PyTorch LSTM Autoencoder (only when torch is available)
# ════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class _LSTMAutoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.2):
            super().__init__()
            drop = dropout if n_layers > 1 else 0.0
            self.enc     = nn.LSTM(input_dim,  hidden_dim, n_layers, batch_first=True, dropout=drop)
            self.dec_rnn = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=drop)
            self.dec_fc  = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            _, (h, _) = self.enc(x)
            z = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
            out, _ = self.dec_rnn(z)
            return self.dec_fc(out)

        @torch.no_grad()
        def recon_error(self, x):
            return ((x - self(x))**2).mean(dim=(1, 2))

        @torch.no_grad()
        def feat_recon_error(self, x):
            return ((x - self(x))**2).mean(dim=1)

else:
    # Pickle compatibility stub — when a .pkl saved with PyTorch backend is
    # loaded in an environment without PyTorch (e.g. Docker without torch),
    # pickle needs the class name _LSTMAutoencoder to exist at module level.
    # LSTMAEDetector.load() handles the actual fallback logic below.
    class _LSTMAutoencoder:
        """Stub for pickle compatibility when PyTorch is unavailable."""
        def __init__(self, *args, **kwargs):
            pass


# ════════════════════════════════════════════════════════════════
# sklearn MLP Autoencoder fallback
# ════════════════════════════════════════════════════════════════

class _MLPAutoencoder:
    """
    Uses sklearn's MLPRegressor as a stable deep autoencoder.
    Architecture: input -> hidden -> latent -> hidden -> input
    """
    def __init__(self, input_dim, epochs=80, lr=1e-3):
        latent = max(8, input_dim // 4)
        hidden = max(32, input_dim // 2)
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden, latent, hidden),
            activation="relu",
            solver="adam",
            learning_rate_init=lr,
            max_iter=epochs,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False,
        )
        self.input_dim = input_dim

    def fit(self, X: np.ndarray):
        self.model.fit(X, X)
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def recon_error(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.reconstruct(X))**2).mean(axis=1)

    def feat_recon_error(self, X: np.ndarray) -> np.ndarray:
        return (X - self.reconstruct(X))**2


# ════════════════════════════════════════════════════════════════
# Unified Detector API
# ════════════════════════════════════════════════════════════════

class LSTMAEDetector:
    def __init__(self, hidden_dim=64, n_layers=2, dropout=0.2,
                 epochs=80, lr=1e-3, batch_size=32):
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.dropout    = dropout
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.scaler     = RobustScaler()
        self.model      = None
        self.threshold  = None
        self.feat_cols  = None
        self.backend    = "torch" if TORCH_AVAILABLE else "sklearn"

    def fit(self, X: np.ndarray, feat_cols: list, labels: np.ndarray = None):
        self.feat_cols = feat_cols
        dim = X.shape[1]

        X_train = self.scaler.fit_transform(X[labels == 0] if labels is not None else X)
        X_all   = self.scaler.transform(X)

        print(f"[AE] backend={self.backend} | "
              f"train_samples={len(X_train)} | epochs={self.epochs}")

        if self.backend == "torch":
            self.model = _LSTMAutoencoder(
                dim, self.hidden_dim, self.n_layers, self.dropout
            ).to(DEVICE)
            opt  = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            crit = nn.MSELoss()
            Xt   = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            dl   = DataLoader(TensorDataset(Xt), batch_size=self.batch_size, shuffle=True)
            self.model.train()
            for ep in range(1, self.epochs + 1):
                ep_loss = 0.0
                for (b,) in dl:
                    opt.zero_grad()
                    loss = crit(self.model(b), b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    ep_loss += loss.item() * len(b)
                if ep % 10 == 0:
                    print(f"  Epoch {ep:>3}/{self.epochs}  loss={ep_loss/len(X_train):.6f}")
            self.model.eval()
            Xt_all = torch.tensor(X_all, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            errs   = self.model.recon_error(Xt_all).cpu().numpy()
        else:
            self.model = _MLPAutoencoder(dim, epochs=self.epochs, lr=self.lr)
            self.model.fit(X_train)
            print(f"  MLP training done. Iterations: {self.model.model.n_iter_}")
            errs = self.model.recon_error(X_all)

        normal_errs    = errs[labels == 0] if labels is not None else errs
        self.threshold = float(np.percentile(normal_errs,50 ))

        # Threshold-anchored score scale:
        #   recon_err = 0           -> score 0.0  (normal)
        #   recon_err = threshold   -> score 0.5  (boundary)
        #   recon_err = 2*threshold -> score 1.0  (anomalous)
        self.score_scale_ = float(2.0 * self.threshold)

        print(f"[AE] Threshold (95th pct of normal): {self.threshold:.6f}")
        print(f"[AE] Score scale (2x threshold):     {self.score_scale_:.6f}")
        return self

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        # Sanitize — streaming can produce NaN from partial event buffers
        X  = np.nan_to_num(X,  nan=0.0, posinf=0.0, neginf=0.0)
        Xs = self.scaler.transform(X)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

        if self.backend == "torch":
            Xt        = torch.tensor(Xs, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            recon_err = self.model.recon_error(Xt).cpu().numpy()
            feat_err  = self.model.feat_recon_error(Xt).cpu().numpy()
        else:
            recon_err = self.model.recon_error(Xs)
            feat_err  = self.model.feat_recon_error(Xs)

        # Threshold-anchored normalization — stable for any batch size
        score = np.clip(recon_err / (self.score_scale_ + 1e-9), 0.0, 1.0)
        pred  = (recon_err > self.threshold).astype(int)

        top_feats = [
            json.dumps([self.feat_cols[i] for i in np.argsort(row)[::-1][:5]])
            for row in feat_err
        ]
        return pd.DataFrame({
            "lstm_recon_error":  np.round(recon_err, 6),
            "lstm_score":        np.round(score, 4),
            "lstm_anomaly":      pred,
            "lstm_top_features": top_feats,
        })

    def save(self, path: str = MODEL_DIR):
        os.makedirs(path, exist_ok=True)
        if self.backend == "torch":
            torch.save(self.model.state_dict(), os.path.join(path, "ae_weights.pt"))
            mdl, self.model = self.model, None
            with open(os.path.join(path, "ae.pkl"), "wb") as f:
                pickle.dump(self, f)
            self.model = mdl
        else:
            with open(os.path.join(path, "ae.pkl"), "wb") as f:
                pickle.dump(self, f)
        print(f"[AE] Saved -> {path}/")

    @classmethod
    def load(cls, path: str = MODEL_DIR):
        with open(os.path.join(path, "ae.pkl"), "rb") as f:
            obj = pickle.load(f)
        if obj.backend == "torch":
            if not TORCH_AVAILABLE:
                # PyTorch not in this environment — retrain with sklearn backend
                print("[AE] PyTorch unavailable. Retraining with sklearn backend...")
                obj.backend = "sklearn"
                dim = len(obj.feat_cols)
                obj.model = _MLPAutoencoder(dim)
                # Load feature store and retrain
                store = pd.read_csv("data/raw/feature_store.csv")
                feat_cols = [c for c in store.columns if c not in ("user_id", "is_anomalous")]
                X = store[feat_cols].fillna(0).to_numpy(dtype=np.float32)
                labels = store["is_anomalous"].astype(int).to_numpy()
                X_train = obj.scaler.transform(X[labels == 0])
                obj.model.fit(X_train)
                print("[AE] sklearn retrain complete.")
                return obj
            obj.model = _LSTMAutoencoder(
                len(obj.feat_cols), obj.hidden_dim, obj.n_layers, obj.dropout
            ).to(DEVICE)
            obj.model.load_state_dict(
                torch.load(os.path.join(path, "ae_weights.pt"), map_location=DEVICE)
            )
            obj.model.eval()
        return obj


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(scores_df, labels):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  precision_score, recall_score, f1_score)
    y, pred, sc = labels.astype(int), scores_df["lstm_anomaly"], scores_df["lstm_score"]
    bk = "PyTorch-LSTM" if TORCH_AVAILABLE else "sklearn-MLP"
    print(f"\n-- Autoencoder ({bk}) Results ------------------------------------")
    print(f"  AUC={roc_auc_score(y,sc):.3f}  "
          f"AP={average_precision_score(y,sc):.3f}  "
          f"P={precision_score(y,pred,zero_division=0):.3f}  "
          f"R={recall_score(y,pred,zero_division=0):.3f}  "
          f"F1={f1_score(y,pred,zero_division=0):.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def train_lstm(feature_store_path="data/raw/feature_store.csv", save=True):
    store     = pd.read_csv(feature_store_path)
    feat_cols = [c for c in store.columns if c not in ("user_id", "is_anomalous")]
    X         = store[feat_cols].fillna(0).to_numpy(dtype=np.float32)
    labels    = store["is_anomalous"].astype(int).to_numpy()

    det    = LSTMAEDetector()
    det.fit(X, feat_cols, labels=labels)
    scores = det.predict(X)
    evaluate(scores, store["is_anomalous"])

    results = pd.concat(
        [store[["user_id", "is_anomalous"]].reset_index(drop=True), scores], axis=1
    )
    if save:
        det.save()
        results.to_csv("data/raw/lstm_scores.csv", index=False)
        print("[AE] Scores -> data/raw/lstm_scores.csv")
    return det, results


if __name__ == "__main__":
    train_lstm()