# Importe de librerias necesarias
import cv2
import numpy as np


# Definicion de la funcion principal
def main():
    # Definicion de variables
    # Color para los bordes de la moneda
    coins_color = (0, 255, 0)  # Color verde BGR
    text_color = (0, 0, 0)  # Color negro BGR

    # Obtener la imagen
    img = cv2.imread('assets/monedas_colombia.jpg', 1)
    # Mostrar la imagen
    cv2.imshow('image', img)
    # Esperar a que se presione una tecla
    cv2.waitKey(0)
    # Cerrar la ventana
    cv2.destroyAllWindows()

    # PROCESO PARA CONTEO DE OBJETOS EN IMAGENES
    # 1. Pasar la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Guardar en imagen en escalada de grises
    cv2.imwrite('assets/monedas_colombia_gray.jpg', gray)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    cv2.imshow('Gauss', blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Guardar imagen con filtro gauss
    cv2.imwrite('assets/monedas_colombia_gauss.jpg', blurred)
    # 3. Use adaptive thresholding
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imshow('Binary', binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Guardar imagen con filtro binario
    cv2.imwrite('assets/monedas_colombia_binary.jpg', binary)
    # Apply morphological operations
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imshow('Closed', closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Guardar imagen con filtro cerrado
    cv2.imwrite('assets/monedas_colombia_closed.jpg', closed)
    # 6. Aplicar Canny
    canny = cv2.Canny(closed, 50, 150)
    cv2.imshow('Canny', canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Guardar imagen con filtro canny
    cv2.imwrite('assets/monedas_colombia_canny.jpg', canny)
    # 7. Encontrar contornos
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 8. Dibujar contornos
    cv2.drawContours(img, contours, -1, coins_color, 2)
    cv2.imshow('Coins', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Guardar imagen con contornos
    cv2.imwrite('assets/monedas_colombia_contours.jpg', img)
    # 9. Contar monedas
    coins = len(contours)
    # 10. Mostrar imagen con monedas marcadas
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f'Coins: {coins}', (10, 30), font, 1, text_color, 2, cv2.LINE_AA)
    cv2.imshow('Coins', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Guardar imagen con monedas marcadas
    cv2.imwrite('assets/monedas_colombia_counts.jpg', img)
    # 11. Crear archivo de texto con el número de monedas
    with open('assets/monedas_colombia_counts.txt', 'w') as file:
        file.write(f'Coins: {coins}')
    # 12. Generar pdf con los resultados

# TODO: Generar un pdf con el algoritmo implementado, los pasos y los resultados obtenidos

# Código para poder ejecutar el archivo
if __name__ == '__main__':
    main()
