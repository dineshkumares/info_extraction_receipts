import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt 
import math 
import itertools
# for making adjacency matrix
import networkx as nx




df = pd.read_csv("test_scratchpart2.csv")
image = cv2.imread("test_scratchpart2.jpg")

print(df)
#for relative distances later
image_height, image_width = image.shape[0], image.shape[1]


"""
This ensures that words are read from top left corner of the image first, 
going line by line from left to right and at last the final bottom right word of the page is read.

Pseudocode:
1) Read words from each line starting from topmost line going towards bottommost line
2) For each word, perform the following:
    - Check words which are in vertical projection with it.
    - Calculate RD_l and RD_r for each of them 
    - Select nearest neighbour words in horizontal direction which have least magnitude of RD_l and RD_r, 
      provided that those words do not have an edge in that direciton.
            - In case, two words have same RD_l or RD_r, the word having higher top coordinate is chosen.
    - Repeat steps from 2.1 to 2.3 similarly for retrieving nearest neighbour words in vertical direction by 
      taking horizontal projection, calculating RD_t and RD_b and choosing words having higher left co-ordinate
      incase of ambiguity
    - Draw edges between word and its 4 nearest neighbours if they are available. 





Line formation:
1) Sort words based on Top coordinate:
2) Form lines as group of words which obeys the following:
    Two words (W_a and W_b) are in same line if:
        Top(W_a) <= Bottom(W_b) and Bottom(W_a) >= Top(W_b)
3) Sort words in each line based on Left coordinate
"""



"""
1) Sort words based on Top coordinate:
"""
#sort df by 'top' coordinate. 
def line_formation(df):
    df.sort_values(by=['ymin'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df)

    """

    _______________y axis__________
    |
    |                       top    
    x axis               ___________________
    |              left | bounding box      |  right
    |                   |___________________|           
    |                       bottom 
    |
    |


    iterate through the rows twice to compare them.
    remember that the axes are inverted.
    """
    master = []
    for idx, row in df.iterrows():

        #flatten the nested list 
        flat_master = list(itertools.chain(*master))
        #print(flat_master)
        #check to see if idx is in flat_master
        if idx not in flat_master:
            top_a = row['ymin']
            bottom_a = row['ymax']
            #top, bottom, right, left
            coordinates = (row['ymin'],row['ymax'],row['xmin'],row['xmax'])
            #print(coordinates)
            line1 = [idx]
            #line1=[row['Object']]
        
            for idx_2, row_2 in df.iterrows():
                #if not the same words
                if not idx == idx_2:
                    top_b = row_2['ymin']
                    bottom_b = row_2['ymax'] 
                    if (top_a <= bottom_b) and (bottom_a >= top_b): 
                        line1.append(idx_2)
            master.append(line1)

    #print(master)

    df2 = pd.DataFrame({'words_indices': master, 'line_number':[x for x in range(1,len(master)+1)]})

    #explode the list columns eg : [1,2,3]
    df2 = df2.set_index('line_number').words_indices.apply(pd.Series).stack()\
            .reset_index(level=0).rename(columns={0:'words_indices'})

    df2['words_indices'] = df2['words_indices'].astype('int')

    #put the line numbers back to the list
    final = df.merge(df2, left_on=df.index, right_on='words_indices')
    final.drop('words_indices', axis=1, inplace=True)

    print(final)

    """
    3) Sort words in each line based on Left coordinate
    """
    final2 =final.sort_values(by=['line_number','xmin'],ascending=True)\
            .groupby('line_number')\
            .head(len(final))\
            .reset_index(drop=True)

    print(final2)

    len(df2)

    return final2

line_formation(df)