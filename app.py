import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io
from scipy.io import wavfile

st.set_page_config(
    page_title="Fourier Series Signal Analyzer",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# JUDUL APLIKASI
# ============================================================
st.title("Fourier Series Signal Analyzer")

st.write(
    "Aplikasi untuk menganalisis pendekatan beberapa jenis sinyal "
    "menggunakan Deret Fourier."
)

# ============================================================
# INPUT JENIS SINYAL (Diperbarui dengan opsi .wav)
# ============================================================
jenis_sinyal = st.selectbox(
    "Pilih Jenis Sinyal",
    [
        "Square Wave",
        "Sawtooth Wave",
        "Triangle Wave",
        "Upload File Audio (.wav)"
    ]
)

# ============================================================
# INPUT JUMLAH HARMONIK
# ============================================================
N = st.slider(
    "Jumlah Harmonik",
    min_value=1,
    max_value=50,
    value=5,
    step=1
)

# ============================================================
# PENGKONDISIAN DOMAIN WAKTU & SINYAL ASLI
# ============================================================
if jenis_sinyal == "Upload File Audio (.wav)":
    
    uploaded_file = st.file_uploader("Pilih file audio (.wav)", type=['wav'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        fs, data = wavfile.read(io.BytesIO(uploaded_file.read()))
        
        if len(data.shape) > 1:
            data = data[:, 0]
            
        N_samples = 2000
        if len(data) > N_samples:
            signal_asli = data[:N_samples]
        else:
            signal_asli = data
            
        if np.max(np.abs(signal_asli)) != 0:
            signal_asli = signal_asli / np.max(np.abs(signal_asli))
            
        t = np.linspace(-np.pi, np.pi, len(signal_asli))
        
    else:
        st.info("👆 Silakan unggah file audio berformat .wav terlebih dahulu.")
        st.stop()

else:
    t = np.linspace(-np.pi, np.pi, 2000)
    
    if jenis_sinyal == "Square Wave":
        signal_asli = np.where(t >= 0, 1, -1)
    elif jenis_sinyal == "Sawtooth Wave":
        signal_asli = t / np.pi
    elif jenis_sinyal == "Triangle Wave":
        signal_asli = (2 / np.pi) * np.arcsin(np.sin(t))

# ============================================================
# PERHITUNGAN KOEFISIEN FOURIER
# ============================================================
a0 = (1 / np.pi) * np.trapezoid(signal_asli, t)

koefisien_an = []
koefisien_bn = []

for n in range(1, N + 1):
    an = (1 / np.pi) * np.trapezoid(signal_asli * np.cos(n * t), t)
    bn = (1 / np.pi) * np.trapezoid(signal_asli * np.sin(n * t), t)
    
    koefisien_an.append(an)
    koefisien_bn.append(bn)

# ============================================================
# REKONSTRUKSI DERET FOURIER
# ============================================================
signal_fourier = np.full_like(t, a0 / 2)

for n in range(1, N + 1):
    signal_fourier += (
        koefisien_an[n - 1] * np.cos(n * t)
        +
        koefisien_bn[n - 1] * np.sin(n * t)
    )

# ============================================================
# MENGHITUNG MSE
# ============================================================
mse = np.mean((signal_asli - signal_fourier) ** 2)
rmse = np.sqrt(mse)

# ============================================================
# INFORMASI HASIL
# ============================================================
st.subheader("Hasil Perhitungan")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Jenis Sinyal", jenis_sinyal)
with col2:
    st.metric("Jumlah Harmonik", N)
with col3:
    st.metric("MSE", f"{mse:.6f}")
with col4:
    st.metric("RMSE", f"{rmse:.6f}")

# ============================================================
# GRAFIK SINYAL
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(t, signal_asli, label="Sinyal Asli")
ax.plot(t, signal_fourier, label="Fourier Series")

ax.set_title(f"Pendekatan {jenis_sinyal} dengan Deret Fourier (N = {N})")
ax.set_xlabel("Waktu (t)")
ax.set_ylabel("Amplitudo")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# ============================================================
# TABEL KOEFISIEN FOURIER
# ============================================================
st.subheader("Koefisien Fourier")
st.write(f"Nilai a₀ = {a0:.6f}")

data_koefisien = {
    "Harmonik (n)": list(range(1, N + 1)),
    "aₙ": [round(x, 6) for x in koefisien_an],
    "bₙ": [round(x, 6) for x in koefisien_bn]
}

st.dataframe(data_koefisien, width="stretch")

# ============================================================
# RUMUS DERET FOURIER
# ============================================================
st.subheader("Rumus Deret Fourier")

st.latex(r"""f(t) = \frac{a_0}{2} + \sum_{n=1}^{N} \left[ a_n\cos(nt) + b_n\sin(nt) \right]""")
st.latex(r"""a_0 = \frac{1}{\pi} \int_{-\pi}^{\pi} f(t)\,dt""")
st.latex(r"""a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(t)\cos(nt)\,dt""")
st.latex(r"""b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(t)\sin(nt)\,dt""")