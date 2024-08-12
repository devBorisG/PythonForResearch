# Amount of required libraries
import cv2
import numpy as np
from pdfrw import PdfWriter, PageMerge, PdfDict, PdfName


# Function to save the image
def save_image(image, name):
    """
    Saves the given image to the 'assets' directory with the specified name.

    Parameters:
    image: The image to be saved.
    name: The name of the file (without extension).
    """
    cv2.imwrite(f'assets/images/{name}.jpg', image)


# Function to show the image
def show_image(image, name):
    """
    Displays the given image in a window with the specified name.

    Parameters:
    image: The image to be displayed.
    name: The name of the window.
    """
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_pdf():
    """
    Generates a PDF with the code and images.
    """
    # Create a new PDF writer
    writer = PdfWriter()

    # Add code to PDF
    with open(__file__, 'r') as code_file:
        code = code_file.read()
        code_page = PageMerge().add(PdfDict(
            Type=PdfName.Page,
            MediaBox=[0, 0, 612, 792],
            Contents=PdfDict(stream=code),
            Resources=PdfDict(
                Font=PdfDict(
                    F1=PdfDict(
                        Type=PdfName.Font,
                        Subtype=PdfName.Type1,
                        BaseFont=PdfName.Helvetica
                    )
                )
            )
        )).render()
        writer.addpage(code_page)

    # Add images to PDF
    images = [
        'monedas_colombia.jpg',
        'monedas_colombia_gray.jpg',
        'monedas_colombia_blurred.jpg',
        'monedas_colombia_binary.jpg',
        'monedas_colombia_closed.jpg',
        'monedas_colombia_canny.jpg',
        'monedas_colombia_coins.jpg',
        'monedas_colombia_coins_count.jpg'
    ]
    for image in images:
        img_page = PageMerge().add(PdfDict(
            Type=PdfName.Page,
            MediaBox=[0, 0, 612, 792],
            Contents=PdfDict(stream=f'<</Type /XObject /Subtype /Image /Width 612 /Height 792 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len(open(f"assets/images/{image}", "rb").read())}>> stream\n{open(f"assets/images/{image}", "rb").read()}\nendstream'),
            Resources=PdfDict(
                XObject=PdfDict(
                    Im0=PdfDict(
                        Type=PdfName.XObject,
                        Subtype=PdfName.Image,
                        Width=612,
                        Height=792,
                        ColorSpace=PdfName.DeviceRGB,
                        BitsPerComponent=8,
                        Filter=PdfName.DCTDecode,
                        Length=len(open(f'assets/images/{image}', 'rb').read())
                    )
                )
            )
        )).render()
        writer.addpage(img_page)

    # Save the PDF
    writer.write('assets/report/monedas_colombia_report.pdf')


# Main function
def main():
    """
    Main function to process the image and count the number of coins.
    """
    # Define colors for drawing
    coins_color = (0, 255, 0)  # Green color for coin edges
    text_color = (0, 0, 0)  # Black color for text

    # Step 1: Load the image
    # Description: Reads the image from the 'assets' directory.
    img = cv2.imread('assets/images/monedas_colombia.jpg', 1)
    show_image(img, 'Original Image')

    # Step 2: Convert to grayscale
    # Description: Converts the loaded image to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image(gray, 'Gray')
    save_image(gray, 'monedas_colombia_gray')

    # Step 3: Apply Gaussian blur
    # Description: Applies Gaussian blur to the grayscale image to reduce noise.
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    show_image(blurred, 'Blurred')
    save_image(blurred, 'monedas_colombia_blurred')

    # Step 4: Adaptive thresholding
    # Description: Uses adaptive thresholding to create a binary image.
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    show_image(binary, 'Binary')
    save_image(binary, 'monedas_colombia_binary')

    # Step 5: Morphological operations
    # Description: Applies morphological operations to close gaps in the binary image.
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    show_image(closed, 'Closed')
    save_image(closed, 'monedas_colombia_closed')

    # Step 6: Canny edge detection
    # Description: Applies Canny edge detection to the morphologically processed image.
    canny = cv2.Canny(closed, 50, 150)
    show_image(canny, 'Canny')
    save_image(canny, 'monedas_colombia_canny')

    # Step 7: Find and draw contours
    # Description: Finds the external contours in the Canny edge-detected image and draws them on the original image.
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, coins_color, 2)
    show_image(img, 'Coins')
    save_image(img, 'monedas_colombia_coins')

    # Step 8: Count coins
    # Description: Counts the number of contours found, which corresponds to the number of coins.
    coins = len(contours)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f'Coins: {coins}', (10, 30), font, 1, text_color, 2, cv2.LINE_AA)
    show_image(img, 'Coins Count')
    save_image(img, 'monedas_colombia_coins_count')

    # Step 9: Save coin count to text file
    # Description: Writes the number of coins to a text file in the 'assets' directory.
    with open('assets/monedas_colombia_counts.txt', 'w') as file:
        file.write(f'Coins: {coins}')

    # Step 10: Generate PDF
    # Description: Generates a PDF with the code and images.
    generate_pdf()


# Code to execute the main function
if __name__ == '__main__':
    main()
