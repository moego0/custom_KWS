"""
Simple Python client SDK for the KWS Trainer API.

Usage:
    pip install requests

    from sdk.kws_api import KWSClient

    client = KWSClient(base_url="http://localhost:8000", api_key="dev-secret")
    client.record(label="keyword")                 # records 5 samples using the server mic
    client.train()                                   # augment + preprocess + train
    model = client.load_model(fmt="h5")             # returns a loaded Keras model
    result = client.predict(r"E:/path/to/test.wav")
    print(result)
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional, Literal, Any, Dict

import requests


class KWSClient:
    """High-level client for interacting with the KWS Trainer API."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url: str = base_url.rstrip("/")
        self.api_key: Optional[str] = api_key

    # --- Internal helpers ---
    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    # --- Core, user-friendly operations ---
    def record(self, label: str, count: int = 5, duration_s: float = 1.0) -> dict:
        """
        Ask the server to record audio from its microphone and save samples.

        Returns server JSON containing saved paths.
        """
        url = f"{self.base_url}/record"
        params = {"label": label, "count": count, "duration": duration_s}
        response = requests.post(url, headers=self._headers(), params=params, timeout=300)
        response.raise_for_status()
        return response.json()

    def train(self) -> dict:
        """
        Full training workflow: augment positives, preprocess ESC-50 negatives, then train.
        Returns server JSON with training summary and model paths.
        """
        # Augment
        r = requests.post(f"{self.base_url}/augment", headers=self._headers(), timeout=600)
        r.raise_for_status()

        # Preprocess negatives
        r = requests.post(f"{self.base_url}/preprocess-esc50", headers=self._headers(), timeout=1800)
        r.raise_for_status()

        # Train
        r = requests.post(f"{self.base_url}/train", headers=self._headers(), timeout=3600)
        r.raise_for_status()
        return r.json()

    def load_model(self, fmt: Literal["h5", "tflite"] = "h5") -> Any:
        """
        Download and load a trained model from the server.
        - fmt='h5': returns a loaded keras.Model (requires tensorflow/keras locally)
        - fmt='tflite': returns a loaded tflite.Interpreter (requires tensorflow locally)
        """
        fmt = "tflite" if fmt == "tflite" else "h5"
        url = f"{self.base_url}/model/{fmt}"

        # Download to a temporary file
        filename = "kws_model.tflite" if fmt == "tflite" else "kws_model.h5"
        with tempfile.TemporaryDirectory() as tmpdir:
            dest_path = os.path.join(tmpdir, filename)
            with requests.get(url, headers=self._headers(), stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)

            if fmt == "h5":
                try:
                    from tensorflow import keras  # type: ignore
                except Exception as exc:  # pylint: disable=broad-except
                    raise RuntimeError("Loading .h5 requires tensorflow/keras installed on client.") from exc
                return keras.models.load_model(dest_path)
            else:
                try:
                    import tensorflow as tf  # type: ignore
                except Exception as exc:  # pylint: disable=broad-except
                    raise RuntimeError("Loading .tflite requires tensorflow installed on client.") from exc
                interpreter = tf.lite.Interpreter(model_path=dest_path)
                interpreter.allocate_tensors()
                return interpreter

    # --- Optional extras mirrored from the API ---
    def status(self) -> dict:
        r = requests.get(f"{self.base_url}/status", timeout=30)
        r.raise_for_status()
        return r.json()

    def upload_sample(self, label: str, wav_path: str) -> dict:
        with open(wav_path, "rb") as f:
            files = {"file": (os.path.basename(wav_path), f, "audio/wav")}
            data = {"label": label}
            r = requests.post(f"{self.base_url}/upload-sample", headers=self._headers(), files=files, data=data, timeout=300)
        r.raise_for_status()
        return r.json()

    def predict(self, wav_path: str) -> dict:
        with open(wav_path, "rb") as f:
            files = {"file": (os.path.basename(wav_path), f, "audio/wav")}
            r = requests.post(f"{self.base_url}/predict", headers=self._headers(), files=files, timeout=60)
        r.raise_for_status()
        return r.json()


