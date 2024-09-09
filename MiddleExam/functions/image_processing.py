import cv2


def process_spectrogram_image(input_image, output_image='modified_spectrogram.png'):
    # Leer la imagen del espectrograma
    img = cv2.imread(input_image)

    # Aplicar procesamiento de imágenes (ejemplo: desenfoque gaussiano)
    processed_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Guardar la imagen procesada
    cv2.imwrite(output_image, processed_img)

    # Mostrar la imagen procesada
    cv2.imshow('Processed Spectrogram', processed_img)
    cv2.waitKey(0)  # Esperar hasta que se presione una tecla
    cv2.destroyAllWindows()

    return processed_img


def dummy():
    print("dummy image processing function")
