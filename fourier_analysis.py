import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACIÓN GENERAL DE LA MUESTREO
# ==========================================
Fs = 1000          # Frecuencia de muestreo (Hz)
T = 1 / Fs         # Periodo de muestreo (s)
L = 2000           # Longitud de la señal (número de muestras)
t = np.arange(0, L) * T  # Vector de tiempo (0 a 2 segundos)

# Frecuencias para la transformada (centrada en 0 Hz)
freqs = np.fft.fftfreq(L, T)
freqs_shifted = np.fft.fftshift(freqs)

# ==========================================
# PASO 1: DEFINICIÓN DE SEÑALES ELEMENTALES
# ==========================================

# 1. Señal Senoidal (50 Hz)
f0 = 50
x_senoidal = np.sin(2 * np.pi * f0 * t)

# 2. Pulso Rectangular (Ancho = 0.2 s, centrado en t = 1.0 s)
x_rectangular = np.where((t >= 0.9) & (t <= 1.1), 1.0, 0.0)

# 3. Función Escalón Unitario (Inicia en t = 0.5 s)
x_escalon = np.where(t >= 0.5, 1.0, 0.0)

# ==========================================
# PASO 2: CÁLCULO DE LA TRANSFORMADA DE FOURIER (FFT)
# ==========================================
def obtener_espectro(x):
    """Calcula la FFT, cambia el cero al centro y obtiene magnitud y fase."""
    X = np.fft.fft(x)
    X_shifted = np.fft.fftshift(X)
    magnitud = np.abs(X_shifted) / L
    fase = np.angle(X_shifted)
    # Filtrar fase para magnitudes casi nulas (evitar ruido numérico)
    fase[magnitud < 1e-3] = 0
    return magnitud, fase

mag_sen, fase_sen = obtener_espectro(x_senoidal)
mag_rec, fase_rec = obtener_espectro(x_rectangular)
mag_esc, fase_esc = obtener_espectro(x_escalon)

# ==========================================
# PASO 3: VISUALIZACIÓN DE RESULTADOS
# ==========================================
fig, axs = plt.subplots(3, 3, figsize=(15, 10))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# --- Fila 1: Senoidal ---
axs[0, 0].plot(t, x_senoidal, 'b')
axs[0, 0].set_title('Senoidal (Tiempo)')
axs[0, 0].set_xlabel('Tiempo (s)')
axs[0, 0].set_ylabel('Amplitud')
axs[0, 0].grid()

axs[0, 1].plot(freqs_shifted, mag_sen, 'r')
axs[0, 1].set_title('Espectro de Magnitud')
axs[0, 1].set_xlim([-100, 100])
axs[0, 1].set_xlabel('Frecuencia (Hz)')
axs[0, 1].set_ylabel('|X(f)|')
axs[0, 1].grid()

axs[0, 2].plot(freqs_shifted, fase_sen, 'g')
axs[0, 2].set_title('Espectro de Fase')
axs[0, 2].set_xlim([-100, 100])
axs[0, 2].set_xlabel('Frecuencia (Hz)')
axs[0, 2].set_ylabel('Fase (rad)')
axs[0, 2].grid()

# --- Fila 2: Pulso Rectangular ---
axs[1, 0].plot(t, x_rectangular, 'b')
axs[1, 0].set_title('Pulso Rectangular (Tiempo)')
axs[1, 0].set_xlabel('Tiempo (s)')
axs[1, 0].set_ylabel('Amplitud')
axs[1, 0].grid()

axs[1, 1].plot(freqs_shifted, mag_rec, 'r')
axs[1, 1].set_title('Espectro de Magnitud (Sinc)')
axs[1, 1].set_xlim([-50, 50])
axs[1, 1].set_xlabel('Frecuencia (Hz)')
axs[1, 1].set_ylabel('|X(f)|')
axs[1, 1].grid()

axs[1, 2].plot(freqs_shifted, fase_rec, 'g')
axs[1, 2].set_title('Espectro de Fase')
axs[1, 2].set_xlim([-50, 50])
axs[1, 2].set_xlabel('Frecuencia (Hz)')
axs[1, 2].set_ylabel('Fase (rad)')
axs[1, 2].grid()

# --- Fila 3: Escalón Unitario ---
axs[2, 0].plot(t, x_escalon, 'b')
axs[2, 0].set_title('Escalón Unitario (Tiempo)')
axs[2, 0].set_xlabel('Tiempo (s)')
axs[2, 0].set_ylabel('Amplitud')
axs[2, 0].grid()

axs[2, 1].plot(freqs_shifted, mag_esc, 'r')
axs[2, 1].set_title('Espectro de Magnitud')
axs[2, 1].set_xlim([-20, 20])
axs[2, 1].set_xlabel('Frecuencia (Hz)')
axs[2, 1].set_ylabel('|X(f)|')
axs[2, 1].grid()

axs[2, 2].plot(freqs_shifted, fase_esc, 'g')
axs[2, 2].set_title('Espectro de Fase')
axs[2, 2].set_xlim([-20, 20])
axs[2, 2].set_xlabel('Frecuencia (Hz)')
axs[2, 2].set_ylabel('Fase (rad)')
axs[2, 2].grid()

plt.tight_layout()
plt.show()

# ==========================================
# PASO 4: VERIFICACIÓN DE PROPIEDADES
# ==========================================

# A. Propiedad de Linealidad: a*x1(t) + b*x2(t) <---> a*X1(f) + b*X2(f)
a, b = 2.0, 0.5
x_lin = a * x_senoidal + b * x_rectangular
X_lin_directa = np.fft.fft(x_lin)
X_lin_propiedad = a * np.fft.fft(x_senoidal) + b * np.fft.fft(x_rectangular)

error_linealidad = np.max(np.abs(X_lin_directa - X_lin_propiedad))
print(f"Error de verificación de Linealidad: {error_linealidad:.2e}")

# B. Desplazamiento en el Tiempo: x(t - t0) <---> X(f) * exp(-j*2*pi*f*t0)
t0 = 0.2  # Desplazamiento de 0.2 segundos
x_desplazada = np.sin(2 * np.pi * f0 * (t - t0))
X_orig = np.fft.fft(x_senoidal)
X_desp_directa = np.fft.fft(x_desplazada)

fase_shift = np.exp(-1j * 2 * np.pi * freqs * t0)
X_desp_propiedad = X_orig * fase_shift

error_desplazamiento = np.max(np.abs(X_desp_directa - X_desp_propiedad))
print(f"Error de verificación de Desplazamiento en Tiempo: {error_desplazamiento:.2e}")