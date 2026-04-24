"""
lstm_detector.py
─────────────────
LSTM Autoencoder for sequential anomaly detection.
Driven by REGISTRY — no hardcoded feature names.

WHY LSTM CATCHES WHAT ISOLATION FOREST MISSES:

  Isolation Forest sees one snapshot and asks:
    "Is THIS single row of 11 metrics unusual?"

  LSTM Autoencoder sees a SEQUENCE and asks:
    "Is this PATTERN over the last 10 minutes unusual?"

  Example — slow memory leak:
    t=0:  mem=500MB  (normal → IF says NORMAL)
    t=1:  mem=520MB  (normal → IF says NORMAL)
    t=2:  mem=540MB  (normal → IF says NORMAL)
    ...
    t=20: mem=900MB  (now IF fires, but you already have a problem)

    LSTM sees [500,520,540,...,900] as a SEQUENCE.
    It was trained on flat/stable sequences.
    This rising pattern has high reconstruction error → LSTM fires EARLY.

HOW THE AUTOENCODER WORKS:

  Training:
    Input sequences of N timesteps → ENCODER compresses to latent vector
    → DECODER reconstructs original sequence
    The model learns: "this is what normal sequences look like"

  Inference:
    Feed current N-step window → get reconstruction
    MSE(input, reconstruction) = anomaly score
    Normal sequence: low MSE (model reconstructs it well)
    Anomalous sequence: high MSE (model can't reconstruct the weird pattern)

  Threshold:
    After training: compute MSE on all training sequences
    threshold = mean(train_MSE) + 3 × std(train_MSE)
    Anything above threshold at inference = ANOMALY

ROLLING BUFFER:
  At inference time you call update_and_predict() every poll interval.
  It appends the new snapshot to a buffer and keeps the last seq_len rows.
  Once the buffer is full, it runs the autoencoder.
  Before that: returns status="WARMING_UP"
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from metric_registry import REGISTRY


# ─────────────────────────────────────────────────────────────────
# PYTORCH MODEL
# ─────────────────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """
    Encoder-Decoder LSTM.

    Encoder:
      (batch, seq_len, n_features)  →  LSTM  →  hidden state (latent)

    Decoder:
      latent repeated seq_len times  →  LSTM  →  linear  →  reconstruction
      (batch, seq_len, n_features)

    The "repeat" trick: since we have no decoder input (we're reconstructing,
    not predicting), we feed the latent vector back to the decoder seq_len
    times. This forces the decoder to rely solely on the latent representation.
    """

    def __init__(self, n_features: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        dropout = 0.1 if n_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,       # (batch, seq, features) not (seq, batch, features)
            dropout=dropout,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) → reconstruction: same shape"""
        batch_size, seq_len, _ = x.shape

        # Encode: last hidden state is the latent representation
        _, (hidden, _) = self.encoder(x)
        latent = hidden[-1]  # (batch, hidden_dim) — take last layer

        # Decode: repeat latent seq_len times as decoder input
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_out, _ = self.decoder(decoder_input)

        return self.output_layer(decoder_out)


# ─────────────────────────────────────────────────────────────────
# HIGH-LEVEL DETECTOR
# ─────────────────────────────────────────────────────────────────
class LSTMDetector:

    def __init__(
            self,
            seq_len: int = 20,
            hidden_dim: int = 64,
            n_layers: int = 2,
            epochs: int = 50,
            batch_size: int = 32,
            lr: float = 1e-3,
            threshold_sigma: float = 3.0,
    ):
        """
        seq_len:
            Number of timesteps in one training/inference window.
            20 steps × 30s per step = 10 minutes of context.
            Longer = catches slower patterns, needs more warmup time.

        hidden_dim:
            Size of the LSTM latent representation.
            64 is good for 11 features. Use 128 for 20+ features.

        threshold_sigma:
            Anomaly threshold = mean_train_error + sigma × std_train_error.
            3.0 = only flag clear outliers (3 standard deviations away).
            2.0 = more sensitive. 4.0 = very conservative.
        """
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.threshold_sigma = threshold_sigma

        self.n_features = len(REGISTRY.feature_names)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[LSTMAutoencoder] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std:  Optional[np.ndarray] = None
        self.threshold:   Optional[float] = None
        self.train_error_stats: Optional[dict] = None
        self.is_fitted = False

        # Rolling buffer for live inference — last seq_len snapshots
        self._buffer: list[np.ndarray] = []

    # ── Helpers ───────────────────────────────────────────────────
    def _scale(self, df: pd.DataFrame) -> np.ndarray:
        X = df[REGISTRY.feature_names].values.astype(np.float32)
        return (X - self.scaler_mean) / (self.scaler_std + 1e-8)

    def _make_windows(self, arr: np.ndarray) -> np.ndarray:
        """
        Sliding window over 2D array.
        (N, features) → (N - seq_len + 1, seq_len, features)

        Example (seq_len=3, simplified to 1 feature):
          [a, b, c, d, e] → [[a,b,c], [b,c,d], [c,d,e]]
        """
        return np.array(
            [arr[i : i + self.seq_len] for i in range(len(arr) - self.seq_len + 1)],
            dtype=np.float32,
        )

    def _mse(self, seq_tensor: torch.Tensor) -> float:
        """Reconstruction error for one sequence."""
        self.model.eval()
        with torch.no_grad():
            rec = self.model(seq_tensor)
            return float(nn.functional.mse_loss(rec, seq_tensor).item())

    # ─────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "LSTMDetector":
        """Train on baseline DataFrame."""
        print(f"\n🧠 Training LSTM Autoencoder")
        print(f"   Rows     : {len(df)}")
        print(f"   Features : {self.n_features}  ({REGISTRY.feature_names})")
        print(f"   seq_len  : {self.seq_len} steps = "
              f"{self.seq_len * 30 // 60}min context")
        print(f"   Hidden   : {self.hidden_dim}")
        print(f"   Epochs   : {self.epochs}")
        print(f"   Device   : {self.device}")

        # Scale
        X_raw = df[REGISTRY.feature_names].values.astype(np.float32)
        self.scaler_mean = X_raw.mean(axis=0)
        self.scaler_std  = X_raw.std(axis=0)
        X_scaled = (X_raw - self.scaler_mean) / (self.scaler_std + 1e-8)

        # Windows
        windows = self._make_windows(X_scaled)
        print(f"   Windows  : {len(windows)}")

        if len(windows) < self.batch_size:
            raise ValueError(
                f"Not enough sequences ({len(windows)}) for batch_size={self.batch_size}. "
                f"Reduce seq_len or get more baseline data."
            )

        tensor_data = torch.tensor(windows).to(self.device)
        loader = DataLoader(
            TensorDataset(tensor_data),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Model
        self.model = LSTMAutoencoder(
            n_features=self.n_features,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
        ).to(self.device)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch,) in loader:
                optimiser.zero_grad()
                rec = self.model(batch)
                loss = criterion(rec, batch)
                loss.backward()
                # Clip gradients — prevents exploding gradients (common in LSTMs)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimiser.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch [{epoch+1:3}/{self.epochs}]  "
                      f"Loss: {total_loss / len(loader):.6f}")

        # Compute threshold from training reconstruction errors
        self.model.eval()
        errors = []
        with torch.no_grad():
            for (batch,) in DataLoader(TensorDataset(tensor_data), batch_size=64):
                rec = self.model(batch)
                per_sample = ((rec - batch) ** 2).mean(dim=(1, 2))
                errors.extend(per_sample.cpu().tolist())

        errors = np.array(errors)
        mean_e = float(np.mean(errors))
        std_e  = float(np.std(errors))
        self.threshold = mean_e + self.threshold_sigma * std_e

        self.train_error_stats = {
            "mean":      round(mean_e, 6),
            "std":       round(std_e, 6),
            "threshold": round(self.threshold, 6),
            "sigma":     self.threshold_sigma,
        }
        self.is_fitted = True

        print(f"\n   Train MSE : mean={mean_e:.6f}  std={std_e:.6f}")
        print(f"   Threshold : {self.threshold:.6f}  "
              f"(mean + {self.threshold_sigma}σ)")
        print("✅ LSTM Autoencoder ready\n")
        return self

    # ─────────────────────────────────────────────────────────────
    def update_and_predict(self, snapshot_df: pd.DataFrame) -> dict:
        """
        Call every poll interval with the latest 1-row snapshot.
        Maintains rolling buffer internally.
        Returns WARMING_UP until seq_len snapshots have been collected.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")

        # Append scaled row to buffer
        row = self._scale(snapshot_df)[0]
        self._buffer.append(row)
        if len(self._buffer) > self.seq_len:
            self._buffer = self._buffer[-self.seq_len:]

        # Not full yet
        if len(self._buffer) < self.seq_len:
            return {
                "detector":    "LSTM",
                "status":      "WARMING_UP",
                "buffer_fill": len(self._buffer),
                "seq_len":     self.seq_len,
                "eta_seconds": (self.seq_len - len(self._buffer)) * 30,
            }

        # Run inference
        seq = np.array(self._buffer, dtype=np.float32)
        seq_tensor = torch.tensor(seq).unsqueeze(0).to(self.device)
        error = self._mse(seq_tensor)

        status = "ANOMALY" if error > self.threshold else "NORMAL"
        severity = float(np.clip(
            (error - self.threshold) / (self.threshold + 1e-8),
            0.0, 1.0
        ))

        return {
            "detector":             "LSTM",
            "status":               status,
            "reconstruction_error": round(error, 6),
            "threshold":            round(self.threshold, 6),
            "severity":             round(severity, 3),
            "train_mse_mean":       self.train_error_stats["mean"],
        }

    # ─────────────────────────────────────────────────────────────
    def save(self, path: str = "models/lstm_detector.pt"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state":       self.model.state_dict(),
            "scaler_mean":       self.scaler_mean,
            "scaler_std":        self.scaler_std,
            "threshold":         self.threshold,
            "train_error_stats": self.train_error_stats,
            "seq_len":           self.seq_len,
            "hidden_dim":        self.hidden_dim,
            "n_layers":          self.n_layers,
            "n_features":        self.n_features,
            "threshold_sigma":   self.threshold_sigma,
        }, path)
        print(f"💾 LSTM saved → {path}")

    @classmethod
    def load(cls, path: str = "models/lstm_detector.pt") -> "LSTMDetector":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        d = cls(
            seq_len=ckpt["seq_len"],
            hidden_dim=ckpt["hidden_dim"],
            n_layers=ckpt["n_layers"],
            threshold_sigma=ckpt["threshold_sigma"],
        )
        d.n_features        = ckpt["n_features"]
        d.scaler_mean       = ckpt["scaler_mean"]
        d.scaler_std        = ckpt["scaler_std"]
        d.threshold         = ckpt["threshold"]
        d.train_error_stats = ckpt["train_error_stats"]
        d.model = LSTMAutoencoder(
            n_features=ckpt["n_features"],
            hidden_dim=ckpt["hidden_dim"],
            n_layers=ckpt["n_layers"],
        )
        d.model.load_state_dict(ckpt["model_state"])
        d.model.eval()
        d.is_fitted = True
        print(f"📂 LSTM loaded ← {path}")
        return d
