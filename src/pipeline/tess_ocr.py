import cv2 
import pytesseract
from pytesseract import Output
import numpy as np
import pandas as pd 

# Need to get 
# (word id, content and bounding box)

image = "../../data/raw/img/339.jpg"
#image = "/Users/udipbohara/Desktop/Datascience_projects/info_extraction_receipts/src/pipeline/test_custom.jpg"
img = cv2.imread(image)


custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(img, config=custom_config)
print(text)

custom_config = r'--oem 3 --psm 6'

d = pytesseract.image_to_data(img, output_type=Output.DICT) #, config=custom_config)
print(d.keys())

#draw boxes 

"""

left is the distance from the upper-left corner of the bounding box, to the left border of the image.
top is the distance from the upper-left corner of the bounding box, to the top border of the image.
width and height are the width and height of the bounding box.
conf is the model's confidence for the prediction for the word within that bounding box.
If conf is -1, that means that the corresponding bounding box contains a block of text, rather than just a single word.

"""
#for graph modeling
xmin,ymin,xmax,ymax,Object = [],[],[],[],[]
df = pd.DataFrame() 

n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) >= 0.74:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        #print(x,y,w,h)
        text = d['text'][i]
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        img = cv2.putText(img, text, (x, y - 1),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1)

        img_height, img_width =  img.shape[0], img.shape[1]
        

  
       
        xmin.append(x)
        ymin.append(y)
        xmax.append(x + w)
        ymax.append(y + h) 
        Object.append(text)
        #print(img_height, img_width)


df['xmin'], df['ymin'], df['xmax'], df['ymax'], df['Object']  = xmin,ymin,xmax,ymax,Object 
df = df[df.Object != " "]

print(df)

# df.to_csv('test550_scratchpart2' + '.csv' ,index = False)
# cv2.imwrite('test550_scratchpart2' + '.jpg', img)

# df.to_csv('test_custom' + '.csv' ,index = False)
# cv2.imwrite('test_custom' + '.jpg', img)

#df.to_csv('../../data/raw/box/test_tess_000' + '.csv' ,index = False)
#cv2.imwrite('../../data/raw/img/test_tess_000' + '.jpg', img)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



