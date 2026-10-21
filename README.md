# Simulación y Análisis de Señales con la Transformada de Fourier

## Descripción del Proyecto
Este proyecto contiene la simulación y análisis de señales en el dominio del tiempo y de la frecuencia utilizando **Python**. Se aplica la **Transformada Rápida de Fourier (FFT)** para estudiar la representación espectral de señales elementales y verificar sus propiedades clave.

---

## Señales Analizadas
1. **Pulso Rectangular:** Visualización de su magnitud espectral en forma de función Sinc.
2. **Función Escalón Unitario:** Análisis de componentes de baja frecuencia.
3. **Función Senoidal ($50 \text{ Hz}$):** Identificación de picos espectrales en la frecuencia fundamental.

---

## Propiedades Verificadas
* **Linealidad:** La FFT de una combinación lineal de señales es igual a la combinación lineal de sus FFTs individuales.
* **Desplazamiento en el Tiempo:** Un retardo en el tiempo produce un cambio de fase lineal en el dominio de la frecuencia.
* **Escalamiento en Frecuencia:** La compresión en el tiempo equivale a una expansión en la frecuencia.

---

## Requisitos e Instalación
Para ejecutar el código localmente, se requiere Python 3.x y las siguientes librerías:

```bash
pip install numpy matplotlib scipy
