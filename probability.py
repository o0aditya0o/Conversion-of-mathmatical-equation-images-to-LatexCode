import pytesseract

def calculate(a, b):
    return a

def prob(img):
    return pytesseract.image_to_string(img, config='-psm 10')