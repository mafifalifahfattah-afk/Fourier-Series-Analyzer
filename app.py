import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

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
# INPUT JENIS SINYAL
# ============================================================

jenis_sinyal = st.selectbox(
    "Pilih Jenis Sinyal",
    [
        "Square Wave",
        "Sawtooth Wave",
        "Triangle Wave"
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
# MEMBUAT DOMAIN WAKTU
# ============================================================

t = np.linspace(-np.pi, np.pi, 2000)


# ============================================================
# MEMBUAT SINYAL ASLI
# ============================================================

if jenis_sinyal == "Square Wave":

    signal_asli = np.where(t >= 0, 1, -1)

elif jenis_sinyal == "Sawtooth Wave":

    signal_asli = t / np.pi

elif jenis_sinyal == "Triangle Wave":

    signal_asli = (2 / np.pi) * np.arcsin(np.sin(t))


# ============================================================
# PERHITUNGAN KOEFISIEN FOURIER
# ============================================================

# Periode:
# T = 2π
#
# Karena:
# omega_0 = 2π / T = 1
#
# Maka:
# a0 = 1/π ∫ f(t) dt
# an = 1/π ∫ f(t) cos(nt) dt
# bn = 1/π ∫ f(t) sin(nt) dt


a0 = (1 / np.pi) * np.trapezoid(signal_asli, t)

koefisien_an = []
koefisien_bn = []


for n in range(1, N + 1):

    an = (1 / np.pi) * np.trapezoid(
        signal_asli * np.cos(n * t),
        t
    )

    bn = (1 / np.pi) * np.trapezoid(
        signal_asli * np.sin(n * t),
        t
    )

    koefisien_an.append(an)
    koefisien_bn.append(bn)


# ============================================================
# REKONSTRUKSI DERET FOURIER
# ============================================================

signal_fourier = np.full_like(
    t,
    a0 / 2
)


for n in range(1, N + 1):

    signal_fourier += (
        koefisien_an[n - 1] * np.cos(n * t)
        +
        koefisien_bn[n - 1] * np.sin(n * t)
    )


# ============================================================
# MENGHITUNG MSE
# ============================================================

mse = np.mean(
    (signal_asli - signal_fourier) ** 2
)

rmse = np.sqrt(mse)

# ============================================================
# INFORMASI HASIL
# ============================================================

st.subheader("Hasil Perhitungan")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Jenis Sinyal",
        jenis_sinyal
    )

with col2:
    st.metric(
        "Jumlah Harmonik",
        N
    )

with col3:
    st.metric(
        "MSE",
        f"{mse:.6f}"
    )

with col4:
    st.metric(
        "RMSE",
        f"{rmse:.6f}"
    )

# ============================================================
# GRAFIK SINYAL
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.plot(
    t,
    signal_asli,
    label="Sinyal Asli"
)

ax.plot(
    t,
    signal_fourier,
    label="Fourier Series"
)

ax.set_title(
    f"Pendekatan {jenis_sinyal} dengan Deret Fourier (N = {N})"
)

ax.set_xlabel("Waktu (t)")
ax.set_ylabel("Amplitudo")

ax.grid(True)
ax.legend()

st.pyplot(fig)


# ============================================================
# TABEL KOEFISIEN FOURIER
# ============================================================

st.subheader("Koefisien Fourier")

st.write(
    f"Nilai a₀ = {a0:.6f}"
)


data_koefisien = {
    "Harmonik (n)": list(range(1, N + 1)),
    "aₙ": [
        round(x, 6)
        for x in koefisien_an
    ],
    "bₙ": [
        round(x, 6)
        for x in koefisien_bn
    ]
}


st.dataframe(
    data_koefisien,
    width="stretch"
)


# ============================================================
# RUMUS DERET FOURIER
# ============================================================

st.subheader("Rumus Deret Fourier")

st.latex(
    r"""
    f(t) =
    \frac{a_0}{2}
    +
    \sum_{n=1}^{N}
    \left[
    a_n\cos(nt)
    +
    b_n\sin(nt)
    \right]
    """
)


st.latex(
    r"""
    a_0 =
    \frac{1}{\pi}
    \int_{-\pi}^{\pi} f(t)\,dt
    """
)


st.latex(
    r"""
    a_n =
    \frac{1}{\pi}
    \int_{-\pi}^{\pi}
    f(t)\cos(nt)\,dt
    """
)


st.latex(
    r"""
    b_n =
    \frac{1}{\pi}
    \int_{-\pi}^{\pi}
    f(t)\sin(nt)\,dt
    """
)


# ============================================================
# KETERANGAN
# ============================================================

st.subheader("Keterangan")

st.write(
    "Koefisien Fourier dihitung secara numerik menggunakan "
    "integrasi pada satu periode sinyal. Kemudian koefisien "
    "tersebut digunakan untuk membentuk pendekatan sinyal "
    "menggunakan Deret Fourier."
)

st.write(
    "Semakin besar jumlah harmonik, umumnya pendekatan "
    "Deret Fourier semakin mendekati sinyal asli."
)