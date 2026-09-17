import numpy as np
import matplotlib.pyplot as plt


# ==============================
# 1. Membuat domain waktu
# ==============================
t = np.linspace(-np.pi, np.pi, 1000)


# ==============================
# 2. Membuat sinyal square wave
# ==============================
signal = np.where(t >= 0, 1, -1)


# ============================
# 3. Menghitung Deret Fourier
# ============================

N = 5

fourier = np.zeros_like(t)

for n in range(1, N + 1, 2):
    fourier += (4 / np.pi) * (1 / n) * np.sin(n * t)


# ============================
# Koefisien Fourier
# ============================

a0 = 0

koefisien_an = []
koefisien_bn = []

for n in range(1, N + 1):
    an = 0

    if n % 2 == 1:
        bn = 4 / (n * np.pi)
    else:
        bn = 0

    koefisien_an.append(an)
    koefisien_bn.append(bn)

print("\nKoefisien Fourier:")
print("a0 =", a0)

for n in range(1, N + 1):
    print(
        f"n={n}: "
        f"a{n}={koefisien_an[n-1]:.4f}, "
        f"b{n}={koefisien_bn[n-1]:.4f}"
    )