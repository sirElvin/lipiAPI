from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from PIL import Image
import cv2
import pytesseract
from matplotlib import pyplot as plt
import numpy as np

app = Flask(__name__)

@app.route('/api', methods = ['POST'])

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Noise removal function
def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1,1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1 )
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return(image)

# revoming border
def remove_borders(image):
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)

# Getting the angle of the skewed image
def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print (len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

def backendhandling():
    if(request.method == 'POST'):
        d = {}
        image_file = request.files['image'] 
        image = cv2.imread(image_file)

        #Pre-processing the image to feed into pytesseract

        # First Straightening up the image

        no_borders = remove_borders(image) ## border removal is necessary before deskewing 
        deskewed_image = deskew(no_borders)

        #adding borders to straighted image
        color = [0,0,0]        
        top, bottom, left, right = [150]*4 #has width of 150 for 4 sides
        image_with_border = cv2.copyMakeBorder(deskewed_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        #grayscaling the image
        gray_image = grayscale(image_with_border)  
        thresh, im_bw = cv2.threshold(gray_image, 200, 230, cv2.THRESH_BINARY)

        #removing noise from image
        no_noise = noise_removal(im_bw)

        #passing the image through page segmentation to identify text from other elements.
        page_segmentation_mode = 6  # Change this to the desired mode
        ocr_result = pytesseract.image_to_string(no_noise, lang="bod", config=f'--psm {page_segmentation_mode}')

        #Using Pytesseract Library
        ocr_result = pytesseract.image_to_string(gray_image, lang="bod")

        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

        translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang="bod_Tibt", tgt_lang='eng_Latn', max_length = 400)

        result = str(translator(ocr_result))
        d['output'] = result
        return d
        
if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
