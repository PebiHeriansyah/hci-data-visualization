"""
app.py — Dashboard Prediksi Kualitas Udara Provinsi Bengkulu

Materi 5: Human-Computer Interaction & Data Visualization
Judul Skripsi: PEMODELAN PREDIKTIF KUALITAS UDARA MENGGUNAKAN RANDOM FOREST DAN
ADAPTIVE NEURO-FUZZY INFERENCE SYSTEM DENGAN OPTIMASI PARTICLE SWARM OPTIMIZATION

Fitur dashboard:
  - Membaca CSV hasil folder 04 secara otomatis, atau menerima upload CSV pengguna.
  - Validasi kolom, tipe, dan nilai kosong dengan pesan error yang informatif.
  - Filter periode tanggal di sidebar.
  - Metrik evaluasi: MAE, RMSE, R² (dihitung ulang dari subset terfilter).
  - Grafik ISPU aktual vs. prediksi (line chart + scatter plot).
  - Tabel detail dan opsi unduhan CSV.
  - Catatan jelas bahwa metrik subset ≠ metrik final keseluruhan test set.

Cara menjalankan:
  streamlit run app.py

Format kolom yang didukung:
  - Wajib: Tanggal (datetime), ISPU_Aktual (float), ISPU_Prediksi (float)
  - Opsional: ISPU_Prediksi_RF (float), ISPU_Prediksi_ANFIS_PSO (float)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Prediksi ISPU — Provinsi Bengkulu",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_FROM_04 = ROOT.parent / "04-machine-learning-ispu" / "output" / "hasil_prediksi_dashboard.csv"
FALLBACK_DEMO_CSV = ROOT / "data" / "hasil_prediksi_sample.csv"

REQUIRED_COLS = ["Tanggal", "ISPU_Aktual", "ISPU_Prediksi"]
OPTIONAL_PRED_COLS = ["ISPU_Prediksi_RF", "ISPU_Prediksi_ANFIS_PSO"]

# ---------------------------------------------------------------------------
# Fungsi utilitas
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_and_validate(path: str | None, uploaded_bytes: bytes | None) -> tuple[pd.DataFrame, str]:
    """Muat dan validasi DataFrame dari path file atau bytes upload.

    Returns:
        (DataFrame tervalidasi, pesan sumber data)

    Raises:
        ValueError: jika skema atau data tidak valid.
    """
    if uploaded_bytes is not None:
        import io
        try:
            df = pd.read_csv(io.BytesIO(uploaded_bytes))
            source = "File yang diunggah pengguna"
        except Exception as exc:
            raise ValueError(f"Gagal membaca file yang diunggah: {exc}") from exc
    elif path and Path(path).exists():
        df = pd.read_csv(path)
        source = f"Otomatis: `{Path(path).name}`"
    else:
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    # Periksa kolom wajib
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: **{missing}**\n\n"
            "Format kolom yang didukung:\n"
            "- `Tanggal` (format tanggal, contoh: `2024-01-15`)\n"
            "- `ISPU_Aktual` (angka desimal)\n"
            "- `ISPU_Prediksi` (angka desimal)\n\n"
            "Kolom opsional: `ISPU_Prediksi_RF`, `ISPU_Prediksi_ANFIS_PSO`"
        )

    # Konversi tipe
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    for col in REQUIRED_COLS[1:] + [c for c in OPTIONAL_PRED_COLS if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=REQUIRED_COLS).sort_values("Tanggal").reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        st.warning(f"⚠️ {n_dropped} baris dibuang karena nilai tidak valid di kolom wajib.")

    if df.empty:
        raise ValueError("Dataset tidak memiliki baris valid setelah validasi.")

    return df, source


def safe_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Hitung MAE, RMSE, R² dengan penanganan edge case."""
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    if len(actual) >= 2 and float(np.var(actual)) > 0:
        r2 = float(r2_score(actual, predicted))
        r2_str = f"{r2:.4f}"
    else:
        r2 = None
        r2_str = "N/A (data tidak cukup atau target konstan)"
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "R2_str": r2_str}


# ---------------------------------------------------------------------------
# UI: Header
# ---------------------------------------------------------------------------
st.title("🌬️ Dashboard Prediksi Kualitas Udara")
st.markdown(
    "**Provinsi Bengkulu** | Perbandingan ISPU Aktual vs. Prediksi  \n"
    "_Materi 5: Human-Computer Interaction & Data Visualization_"
)

# ---------------------------------------------------------------------------
# Sidebar: sumber data
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Sumber Data")
    uploaded_file = st.file_uploader(
        "Upload CSV prediksi Anda",
        type=["csv"],
        help=(
            "Kolom wajib: Tanggal, ISPU_Aktual, ISPU_Prediksi\n"
            "Kolom opsional: ISPU_Prediksi_RF, ISPU_Prediksi_ANFIS_PSO"
        ),
    )
    st.markdown("---")
    st.caption(
        "Jika tidak ada file yang diunggah, dashboard mencoba membaca hasil "
        "dari folder `04-machine-learning-ispu/output/`, lalu fallback ke data demo."
    )

# ---------------------------------------------------------------------------
# Muat data
# ---------------------------------------------------------------------------
data: pd.DataFrame | None = None
data_source: str = ""
load_error: str = ""

if uploaded_file is not None:
    try:
        with st.spinner("Memuat file yang diunggah..."):
            data, data_source = load_and_validate(None, uploaded_file.getvalue())
    except Exception as exc:
        load_error = str(exc)
else:
    # Coba hasil folder 04, lalu fallback ke demo
    for candidate in [str(DEFAULT_CSV_FROM_04), str(FALLBACK_DEMO_CSV)]:
        try:
            with st.spinner(f"Mencari data di `{Path(candidate).name}`..."):
                data, data_source = load_and_validate(candidate, None)
            break
        except (FileNotFoundError, ValueError):
            continue
    else:
        load_error = (
            "Tidak ada sumber data yang dapat dimuat. "
            "Upload CSV atau jalankan pipeline folder 04 terlebih dahulu:\n\n"
            "```bash\n"
            "cd ../04-machine-learning-ispu\n"
            "python src/pipeline.py\n"
            "```"
        )

if load_error:
    st.error(f"**Data tidak dapat dimuat:**\n\n{load_error}")
    st.stop()

# Tampilkan banner sumber data dan peringatan sintetis
if "sintetis" in data_source.lower() or "sample" in data_source.lower():
    st.info(
        f"📊 **Sumber data**: {data_source}  \n"
        "⚠️ **Data ini SINTETIS.** Jangan gunakan untuk pelaporan penelitian. "
        "Ganti dengan hasil model dari data ISPU resmi."
    )
else:
    st.success(f"📊 **Sumber data**: {data_source}")

# ---------------------------------------------------------------------------
# Sidebar: filter periode
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🗓️ Filter Periode")
    start_date = data["Tanggal"].dt.date.min()
    end_date = data["Tanggal"].dt.date.max()
    selected_range = st.date_input(
        "Rentang tanggal",
        value=(start_date, end_date),
        min_value=start_date,
        max_value=end_date,
    )

    # Pilih model prediksi jika ada kolom opsional
    available_preds = ["ISPU_Prediksi"] + [c for c in OPTIONAL_PRED_COLS if c in data.columns]
    pred_label_map = {
        "ISPU_Prediksi": "ANFIS-PSO (atau default)",
        "ISPU_Prediksi_RF": "Random Forest",
        "ISPU_Prediksi_ANFIS_PSO": "ANFIS-PSO",
    }
    if len(available_preds) > 1:
        selected_pred = st.selectbox(
            "Kolom prediksi untuk metrik",
            options=available_preds,
            format_func=lambda c: pred_label_map.get(c, c),
        )
    else:
        selected_pred = "ISPU_Prediksi"

# ---------------------------------------------------------------------------
# Filter data berdasarkan rentang tanggal
# ---------------------------------------------------------------------------
if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
    first_day, last_day = selected_range
elif isinstance(selected_range, (tuple, list)) and len(selected_range) == 1:
    first_day = last_day = selected_range[0]
elif isinstance(selected_range, (tuple, list)) and len(selected_range) == 0:
    first_day, last_day = start_date, end_date
else:
    first_day = last_day = selected_range

if first_day > last_day:
    st.sidebar.error("Tanggal awal harus sebelum atau sama dengan tanggal akhir.")
    st.stop()

view = data[
    (data["Tanggal"].dt.date >= first_day) & (data["Tanggal"].dt.date <= last_day)
].copy()

if view.empty:
    st.warning("Tidak ada data pada periode terpilih. Perluas rentang tanggal.")
    st.stop()

# ---------------------------------------------------------------------------
# Metrik evaluasi
# ---------------------------------------------------------------------------
actual_arr = view["ISPU_Aktual"].to_numpy(dtype=float)
pred_col = selected_pred if selected_pred in view.columns else "ISPU_Prediksi"
predicted_arr = view[pred_col].to_numpy(dtype=float)

# Buang NaN berpasangan
valid_mask = np.isfinite(actual_arr) & np.isfinite(predicted_arr)
actual_valid = actual_arr[valid_mask]
predicted_valid = predicted_arr[valid_mask]

if len(actual_valid) == 0:
    st.error("Tidak ada pasangan nilai aktual-prediksi yang valid dalam periode ini.")
    st.stop()

metrics = safe_metrics(actual_valid, predicted_valid)

st.subheader("📈 Ringkasan Periode Terpilih")
st.caption(
    "⚠️ Metrik di bawah dihitung dari **subset periode yang dipilih**, "
    "bukan dari keseluruhan test set. Metrik evaluasi final penelitian "
    "harus dihitung dari test set yang telah ditetapkan di metodologi."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Pengamatan", f"{len(view):,}")
col2.metric("MAE", f"{metrics['MAE']:.2f}")
col3.metric("RMSE", f"{metrics['RMSE']:.2f}")
col4.metric("R²", metrics["R2_str"] if metrics["R2"] is None else f"{metrics['R2']:.4f}")

# ---------------------------------------------------------------------------
# Grafik 1: Line chart ISPU aktual vs prediksi
# ---------------------------------------------------------------------------
st.subheader("📉 Tren ISPU dari Waktu ke Waktu")

# Tampilkan semua kolom prediksi yang tersedia
melt_vars = ["ISPU_Aktual", pred_col]
melt_vars += [c for c in OPTIONAL_PRED_COLS if c in view.columns and c != pred_col]
melt_vars = list(dict.fromkeys(melt_vars))  # deduplicate, jaga urutan

long = view.melt(
    id_vars="Tanggal",
    value_vars=melt_vars,
    var_name="Seri",
    value_name="Nilai ISPU",
)
label_map = {
    "ISPU_Aktual": "ISPU Aktual",
    "ISPU_Prediksi": "Prediksi (ANFIS-PSO/default)",
    "ISPU_Prediksi_RF": "Prediksi RF",
    "ISPU_Prediksi_ANFIS_PSO": "Prediksi ANFIS-PSO",
}
long["Keterangan"] = long["Seri"].map(label_map).fillna(long["Seri"])

fig_line = px.line(
    long,
    x="Tanggal",
    y="Nilai ISPU",
    color="Keterangan",
    markers=True,
    labels={"Tanggal": "Tanggal Pengamatan", "Nilai ISPU": "Nilai ISPU", "Keterangan": ""},
    color_discrete_map={
        "ISPU Aktual": "#1f77b4",
        "Prediksi (ANFIS-PSO/default)": "#ff7f0e",
        "Prediksi RF": "#2ca02c",
        "Prediksi ANFIS-PSO": "#ff7f0e",
    },
)
fig_line.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_line, use_container_width=True)

# ---------------------------------------------------------------------------
# Grafik 2: Scatter aktual vs prediksi
# ---------------------------------------------------------------------------
st.subheader("🎯 Aktual vs. Prediksi")
col_sc1, col_sc2 = st.columns([2, 1])

with col_sc1:
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=actual_valid,
        y=predicted_valid,
        mode="markers",
        name=pred_label_map.get(pred_col, pred_col),
        marker=dict(size=8, opacity=0.65, color="#ff7f0e"),
    ))
    lo = float(min(actual_valid.min(), predicted_valid.min()))
    hi = float(max(actual_valid.max(), predicted_valid.max()))
    if lo == hi:
        hi = lo + 1.0
    fig_scatter.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        name="Ideal (prediksi = aktual)",
        line=dict(dash="dash", color="gray", width=1),
    ))
    fig_scatter.update_layout(
        xaxis_title="ISPU Aktual",
        yaxis_title="ISPU Prediksi",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption("Titik mendekati garis putus-putus = prediksi mendekati nilai aktual.")

with col_sc2:
    st.markdown("**Distribusi Kesalahan**")
    errors = predicted_valid - actual_valid
    fig_hist = px.histogram(
        pd.DataFrame({"Kesalahan": errors}),
        x="Kesalahan",
        nbins=20,
        color_discrete_sequence=["#1f77b4"],
        labels={"Kesalahan": "Kesalahan (Prediksi − Aktual)"},
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
    fig_hist.update_layout(yaxis_title="Frekuensi", showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("Garis merah = tidak ada kesalahan. Distribusi simetris di sekitar 0 lebih baik.")

# ---------------------------------------------------------------------------
# Tabel detail dan download
# ---------------------------------------------------------------------------
with st.expander("📋 Lihat Data Terperinci"):
    display_cols = ["Tanggal", "ISPU_Aktual", pred_col]
    display_cols += [c for c in OPTIONAL_PRED_COLS if c in view.columns and c not in display_cols]
    st.dataframe(
        view[display_cols].style.format(
            {col: "{:.2f}" for col in display_cols if col != "Tanggal"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    csv_bytes = view[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Unduh Data Periode Ini (CSV)",
        data=csv_bytes,
        file_name="ispu_prediksi_periode_terpilih.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Dashboard ini adalah implementasi materi Human-Computer Interaction & Data Visualization. "
    "Algoritma inti (RF, ANFIS, PSO) berada di folder `04-machine-learning-ispu/`. "
    "Data sintetis hanya untuk demonstrasi alur; ganti dengan data tervalidasi untuk penelitian."
)
