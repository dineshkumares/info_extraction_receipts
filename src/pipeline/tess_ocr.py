import cv2 
import pytesseract
from pytesseract import Output
import numpy as np


# Need to get 
# (word id, content and bounding box)

image = "../../data/raw/img/004.jpg"
img = cv2.imread(image)


custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(img, config=custom_config)
print(text)

custom_config = r'--oem 3 --psm 6'

d = pytesseract.image_to_data(img, output_type=Output.DICT) #, config=custom_config)
print(d.keys())

#draw boxes 
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) >= 0.74:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        text = d['text'][i]
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img = cv2.putText(img, text, (x, y - 1),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



