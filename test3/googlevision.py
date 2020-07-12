
import os, io
from google.cloud import vision
from google.cloud.vision import types
import pandas as pd 

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'../ServiceAccountToken.json'


client = vision.ImageAnnotatorClient() 

image = 'data/raw/img/004.jpg'

with io.open(image,'rb') as image_file:
    content = image_file.read() 

image = vision.types.Image(content=content)
response = client.text_detection(image=image)  # returns TextAnnotation
df = pd.DataFrame(columns=['locale', 'description'])

texts = response.text_annotations
for text in texts:
    df = df.append(
        dict(
            locale=text.locale,
            description=text.description
        ),
        ignore_index=True
    )