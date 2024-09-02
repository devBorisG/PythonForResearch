# TODO: 1. Grabar un ejemplo de nota de voz
# TODO: 2. Filtrar los intervalos de silencio
# TODO: 3. Muestreo, cuantificación y codificación
# TODO: 4. Transformacion de Fourier
# TODO: 5. Visualizar el Histograma de frecuencias
# TODO: 6. Analizar y Reporte de resultados

# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Leer la nota de voz en formato WAV
data, samplerate = sf.read('./assets/Grabacion.wav')

# Reproducir la nota de voz
sd.play(data, samplerate)
status = sd.wait()

# Graficar la nota de voz
plt.plot(data)
plt.title('Nota de voz')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.show()

# Eliminar los intervalos de silencio
umbral = 0.01
data_filtrada = data[data > umbral]