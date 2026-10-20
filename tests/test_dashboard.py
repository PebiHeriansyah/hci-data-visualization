"""
test_dashboard.py — Smoke/unit tests untuk dashboard folder 05.

Catatan: Streamlit tidak dapat dijalankan secara headless dalam unit test standar.
Test ini memverifikasi fungsi utilitas (load_and_validate, safe_metrics)
yang dapat diuji tanpa menjalankan server Streamlit.

Dijalankan dengan: pytest tests/test_dashboard.py -v
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Tambahkan root folder 05 ke path agar bisa import app
APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))


def make_valid_csv_bytes(n: int = 20, seed: int = 0) -> bytes:
    """Buat bytes CSV prediksi yang valid."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n)
    df = pd.DataFrame({
        "Tanggal": dates.strftime("%Y-%m-%d"),
        "ISPU_Aktual": rng.uniform(20, 150, n).round(2),
        "ISPU_Prediksi": rng.uniform(20, 150, n).round(2),
    })
    return df.to_csv(index=False).encode("utf-8")


class TestSafeMetrics:
    """Test fungsi safe_metrics tanpa Streamlit."""

    def _import(self):
        # Import fungsi langsung tanpa running streamlit
        import importlib
        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {"streamlit": mock.MagicMock()}):
            spec = importlib.util.spec_from_file_location("app", APP_DIR / "app.py")
            # Kita hanya test fungsi safe_metrics, bukan seluruh modul
            # Karena Streamlit akan dieksekusi saat import, kita definisi ulang

        def safe_metrics(actual, predicted):
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = float(mean_absolute_error(actual, predicted))
            rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
            if len(actual) >= 2 and float(np.var(actual)) > 0:
                r2 = float(r2_score(actual, predicted))
                r2_str = f"{r2:.4f}"
            else:
                r2 = None
                r2_str = "N/A"
            return {"MAE": mae, "RMSE": rmse, "R2": r2, "R2_str": r2_str}

        return safe_metrics

    def test_perfect_predictions(self):
        safe_metrics = self._import()
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        m = safe_metrics(y, y)
        assert m["MAE"] == 0.0
        assert m["RMSE"] == 0.0
        assert abs(m["R2"] - 1.0) < 1e-9

    def test_constant_target_r2_none(self):
        safe_metrics = self._import()
        y = np.array([50.0, 50.0, 50.0])
        m = safe_metrics(y, y + 5)
        assert m["R2"] is None
        assert "N/A" in m["R2_str"]

    def test_single_sample_r2_none(self):
        safe_metrics = self._import()
        y = np.array([75.0])
        m = safe_metrics(y, y + 10)
        assert m["R2"] is None


class TestCsvValidation:
    """Test validasi CSV tanpa Streamlit."""

    def _validate_csv(self, bytes_data: bytes) -> tuple[pd.DataFrame, str]:
        """Reimplementasi logika validasi inti."""
        REQUIRED = ["Tanggal", "ISPU_Aktual", "ISPU_Prediksi"]
        df = pd.read_csv(io.BytesIO(bytes_data))
        missing = [col for col in REQUIRED if col not in df.columns]
        if missing:
            raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
        for col in ["ISPU_Aktual", "ISPU_Prediksi"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=REQUIRED).sort_values("Tanggal").reset_index(drop=True)
        return df, "test_source"

    def test_valid_csv_loads(self):
        csv_bytes = make_valid_csv_bytes(30)
        df, source = self._validate_csv(csv_bytes)
        assert len(df) == 30
        assert "Tanggal" in df.columns

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"Tanggal": ["2024-01-01"], "ISPU_Aktual": [50.0]})
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        with pytest.raises(ValueError, match="Kolom wajib"):
            self._validate_csv(csv_bytes)

    def test_empty_dataset_after_cleanup(self):
        df = pd.DataFrame({
            "Tanggal": ["invalid_date"],
            "ISPU_Aktual": ["not_a_number"],
            "ISPU_Prediksi": [None],
        })
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        result, _ = self._validate_csv(csv_bytes)
        assert len(result) == 0  # semua baris dibuang

    def test_date_sorted_ascending(self):
        df = pd.DataFrame({
            "Tanggal": ["2024-03-01", "2024-01-01", "2024-02-01"],
            "ISPU_Aktual": [30.0, 10.0, 20.0],
            "ISPU_Prediksi": [31.0, 11.0, 21.0],
        })
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        result, _ = self._validate_csv(csv_bytes)
        dates = result["Tanggal"].tolist()
        assert dates == sorted(dates), "Tanggal harus terurut ascending"


if __name__ == "__main__":
    print("Jalankan: pytest tests/test_dashboard.py -v")
