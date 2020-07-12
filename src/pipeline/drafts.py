

"""
from flair.embeddings import BytePairEmbeddings
from flair.data import Sentence

embedding = BytePairEmbeddings('en')


sentence = Sentence('ten')

sentence = Sentence('RECEIPT #: CSP0393921')
sentence = Sentence('NO. 17-G, JALAN SETIA INDAH')
embedding.embed(sentence)


for token in sentence:
    print(token)
    print(token.embedding)
    print(token.embedding.shape)
"""

import re
# regex for date. The pattern in the receipt is in 30.07.2007 in DD:MM:YYYY

extracted_text = '09/12/1993'
date_pattern = r'(0[1-9]|[12][0-9]|3[01])[.](0[1-9]|1[012])[.](19|20)\d\d'
date = re.search(date_pattern, extracted_text)
#receipt_ocr['date'] = date
print(date)

