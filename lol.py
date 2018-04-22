from PIL import Image
import pytesseract

img = Image.open("F:\CODING\ProjectLatex\dataset\Sample085\img085-000019.png")

print(pytesseract.image_to_string(img,config='-psm 8'))