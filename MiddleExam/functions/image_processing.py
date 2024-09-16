# MiddleExam/functions/image_processing.py
import cv2
import numpy as np

def process_spectrogram_image(input_image, output_image='modified_spectrogram.png'):
    # Leer la imagen
    img = cv2.imread(input_image)

    # Aplicar un desenfoque gaussiano
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Aplicar un filtro de nitidez (sharpening)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(blurred_img, -1, kernel)

    # Detectar bordes usando el algoritmo de Canny
    edges = cv2.Canny(sharpened_img, 100, 200)

    # Añadir ruido gaussiano
    row, col, ch = img.shape
    mean = 0
    sigma = 0.05  # Reducir la cantidad de ruido
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_img = sharpened_img + gauss

    # Guardar la imagen procesada
    cv2.imwrite(output_image, noisy_img)

    return noisy_img