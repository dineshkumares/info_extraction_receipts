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

#print(df)
#for relative distances later
image_height, image_width = image.shape[0], image.shape[1]


"""

 
Line formation:
1) Sort words based on Top coordinate:
2) Form lines as group of words which obeys the following:
    Two words (W_a and W_b) are in same line if:
        Top(W_a) <= Bottom(W_b) and Bottom(W_a) >= Top(W_b)
3) Sort words in each line based on Left coordinate

This ensures that words are read from top left corner of the image first, 
going line by line from left to right and at last the final bottom right word of the page is read.

"""



"""
1) Sort words based on Top coordinate:
"""
#sort df by 'top' coordinate. 
def line_formation(df):
    df.sort_values(by=['ymin'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    #print(df)

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

            #line will atleast have the word in it
            line = [idx]
            #line1=[row['Object']]
        
            for idx_2, row_2 in df.iterrows():
                #if not the same words
                if not idx == idx_2:
                    top_b = row_2['ymin']
                    bottom_b = row_2['ymax'] 
                    if (top_a <= bottom_b) and (bottom_a >= top_b): 
                        line.append(idx_2)
            master.append(line)

    #print(master)

    df2 = pd.DataFrame({'words_indices': master, 'line_number':[x for x in range(1,len(master)+1)]})

   
    #explode the list columns eg : [1,2,3]
    df2 = df2.set_index('line_number').words_indices.apply(pd.Series).stack()\
            .reset_index(level=0).rename(columns={0:'words_indices'})

    df2['words_indices'] = df2['words_indices'].astype('int')

    
    #put the line numbers back to the list
    final = df.merge(df2, left_on=df.index, right_on='words_indices')
    final.drop('words_indices', axis=1, inplace=True)

    #print(final)

    """
    3) Sort words in each line based on Left coordinate
    """
    final2 =final.sort_values(by=['line_number','xmin'],ascending=True)\
            .groupby('line_number')\
            .head(len(final))\
            .reset_index(drop=True)

    #print(final2)

    #len(df2)

    return final2

line_formation(df)



"""
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

"""

df = line_formation(df)
#print(df)
def grapher(df):

    #horizontal edges formation
    print(df)
    df.reset_index(inplace=True)
    horizontal_dict = {}            
    
   
    grouped = df.groupby('line_number')
    
    #for undirected graph construction
    connections = {}

    #left
    dict_2 = {}

    #right
    hori2 = {}

    for _,group in grouped:
        a = group['index'].tolist()
        b = group['index'].tolist()
        #b.reverse()
        connection = {a[i]:[a[i+1]] for i in range(len(a)-1) }

        my_dict = {a[i]:{'right':[a[i+1]]} for i in range(len(a)-1) }
        my_dict2 = {b[i+1]:{'left':[b[i]]} for i in range(len(b)-1) }

        # for r_idx2, row2 in group.iterrows():
        #     distance_x1 = row['xmax']
        #     for r_idx, row in group.iterrows():
        #         if r_idx != r_idx2:
        #             distance_x2 = row2['xmin']
        #             relative_distance = (distance_x1-distance_x2)/image_width
                    

        horizontal_dict.update(my_dict)
        hori2.update(my_dict2)
        connections.update(connection)
    dic1,dic2 = horizontal_dict, hori2
    
    print(connections)
    result = {}
    for key in (dic1.keys() | dic2.keys()):
        if key in dic1: result.setdefault(key, []).append(dic1[key])
        if key in dic2: result.setdefault(key, []).append(dic2[key])
    print(result)

    
    print()
    result = {}
    for key in (dic1.keys() | dic2.keys()):
        if key in dic1: result.setdefault(key, {}).update(dic1[key])
        if key in dic2: result.setdefault(key, {}).update(dic2[key])

    print(result)
    #get_numeric = lambda x: x if x != UNDEFINED else 0
    print(result[0]['right'][0])
    print(result[1]['right'][0])
    print(result[1]['left'][0])

    for k,v in horizontal_dict.items():
        #df k get values xmin
        #df v get values xmax
        #get distance and push it into a list
        #use that for features


        pass
        #horizontal_dict[k].append('asd')
     
    G = nx.from_dict_of_lists(connections)
    layout = nx.spring_layout(G)
    nx.draw(G, layout, with_labels=True)
    plt.show()
grapher(df)





"""
        for r_idx, row in group.iloc[:-1].iterrows():
            horizontal_dict[r_idx] = [r_idx + 1] 
            distance_x1 = row['xmax']
            for r_idx2, row2 in group.iloc[:-1].iterrows():
                if r_idx != r_idx2:
                    distance_x2 = row2['xmin']
                    relative_distance = (distance_x1-distance_x2)/image_width
                    dict_2[r_idx] = {'right':(r_idx2,float('{:.2f}'.format(relative_distance)))}

    print(horizontal_dict)
    print(dict_2)
    G = nx.from_dict_of_lists(horizontal_dict)
    layout = nx.spring_layout(G)
    nx.draw(G, layout, with_labels=True)
    plt.show()
    
grapher(df)


{0: [1], 1: [2], 2: [6], 
3: [4], 4: [5], 5: [6], 6: [7], 
7: [8], 8: [15], 9: [10], 10: [11], 
11: [12], 12: [13], 13: [14], 14: [15], 
15: [23], 16: [17], 17: [19], 18: [19], 
19: [20], 20: [23], 21: [23], 22: [23],
23: [24], 24: [51], 25: [26], 26: [27], 
27: [28], 28: [31], 29: [30], 30: [31], 
31: [32], 32: [70], 33: [34], 34: [36],
35: [38], 36: [37], 37: [40], 38: [39], 
39: [40], 40: [41], 41: [42], 42: [45], 
43: [44], 44: [45], 45: [56], 46: [47], 
47: [48], 48: [49], 49: [50], 50: [51], 
51: [70], 52: [53], 53: [54], 54: [55], 
55: [56], 56: [59], 57: [58], 58: [66], 
59: [63], 60: [61], 61: [62], 62: [63], 
63: [67], 64: [65], 65: [66], 66: [67], 
67: [70], 68: [70], 69: [70], 70: [75],
71: [72], 72: [73], 73: [74], 74: [75], 
75: [70], 76: [77], 77: [78], 78: [80], 
79: [80], 80: [81], 81: [82]}
"""