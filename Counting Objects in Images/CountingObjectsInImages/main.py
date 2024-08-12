# Importe de librerias necesarias
import cv2


# Definicion de la funcion principal
def main():
    print('Hello, World!')


# Obtener la imagen
img = cv2.imread('assets/monedas_colombia.jpg', 1)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# TODO: Obtener la imagen
# TODO: Procesarla para obtener las monedas
# TODO: Contar las monedas
# TODO: Mostrar la imagen con las monedas marcadas
# TODO: Mostrar el número de monedas
# TODO: Guardar la imagen con las monedas marcadas
# TODO: Crear un archivo de texto con el número de monedas
# TODO: Generar un pdf con el algoritmo implementado, los pasos y los resultados obtenidos

# Código para poder ejecutar el archivo
if __name__ == '__main__':
    main()
