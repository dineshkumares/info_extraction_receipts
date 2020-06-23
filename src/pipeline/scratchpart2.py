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

#df = pd.read_csv("../../data/raw/box/test_tess_339.csv")
#image = cv2.imread('../../data/raw/img/test_tess_339.jpg')
# print(df)

#df = pd.read_csv("test550_scratchpart2.csv")
#image = cv2.imread('test550_scratchpart2.jpg')
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
    #further cleaning
    df.dropna(inplace=True)
    #sort from top to bottom
    df.sort_values(by=['ymin'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    #print(df)
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
            #coordinates = (row['ymin'],row['ymax'],row['xmin'],row['xmax'])
            #print(coordinates)

            #line will atleast have the word in it
            line = [idx]
            #line1=[row['Object']]
        
            for idx_2, row_2 in df.iterrows():
                #check to see if idx_2 is in flat_master removes ambiguity
                #picks higher cordinate one. 
                if idx_2 not in flat_master:
                #if not the same words
                    if not idx == idx_2:
                        top_b = row_2['ymin']
                        bottom_b = row_2['ymax'] 
                        if (top_a <= bottom_b) and (bottom_a >= top_b): 
                            line.append(idx_2)
                            #print(line)
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
   #print(final2)
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
def grapher(df, show=False):

    #horizontal edges formation
    #print(df)
    df.reset_index(inplace=True)
            
    
   
    grouped = df.groupby('line_number')
    
    #for undirected graph construction
    horizontal_connections = {}

    #left
    left_connections = {}    

    #right
    right_connections = {}

    for _,group in grouped:
        a = group['index'].tolist()
        b = group['index'].tolist()
        #b.reverse()
        #a = 0,1,2
        #2
        horizontal_connection = {a[i]:a[i+1] for i in range(len(a)-1) }

        #storing directional connections
        right_dict_temp = {a[i]:{'right':a[i+1]} for i in range(len(a)-1) }
        left_dict_temp = {b[i+1]:{'left':b[i]} for i in range(len(b)-1) }


        #add the indices in the dataframes
        for i in range(len(a)-1):
           df.loc[df['index'] == a[i], 'right'] = int(a[i+1])
           df.loc[df['index'] == a[i+1], 'left'] = int(a[i])
    
        left_connections.update(right_dict_temp)
        right_connections.update(left_dict_temp)
        horizontal_connections.update(horizontal_connection)


    dic1,dic2 = left_connections, right_connections
    
    #print(df)

    #this can be used to update the connections dictionary
    # result = {}
    # for key in (dic1.keys() | dic2.keys()):
    #     if key in dic1: result.setdefault(key, []).append(dic1[key])
    #     if key in dic2: result.setdefault(key, []).append(dic2[key])
    # #print(result)




    
    # print()
    # result = {}
    # for key in (dic1.keys() | dic2.keys()):
    #     if key in dic1: result.setdefault(key, {}).update(dic1[key])
    #     if key in dic2: result.setdefault(key, {}).update(dic2[key])

 
    #get_numeric = lambda x: x if x != UNDEFINED else 0
    # print(result[0]['right'][0])
    # print(result[1]['right'][0])
    # print(result[1]['left'][0])

    #add it to the dataframe. 
  


    #append right and left incides to the main df for features calculation
    # source_dict = left_connections
    # for indices in source_dict.keys():
    #     for ticker in source_dict[indices].keys():
    #         right_item_index = source_dict[indices][ticker]
    #         df.loc[df['index'] == indices, 'right'] = int(right_item_index)

    # source_dict = right_connections
    # for indices in source_dict.keys():
    #     for ticker in source_dict[indices].keys():
    #         left_item_index = source_dict[indices][ticker]
    #         df.loc[df['index'] == indices, 'left'] = int(left_item_index)


    


    # flatten the dictionary


    #print(pd.DataFrame(reform))
   
   
   
    #RD_l and RD_t are negative

        #print(v)
        #RD_r = (df1['xmax'] - df2['xmin'])/image_width
        #print(RD_r)
        #a[k]:[a[k+1]]
        

        #try except

        #print(df1)
        
        #df k get values xmin
        #df v get values xmax
        #get distance and push it into a list
        #use that for features


    
    #verticle connections formation


    bottom_connections = {}
    top_connections = {}

    for idx, row in df.iterrows():
        if idx not in bottom_connections.keys():
            #below = False 
            right_a = row['xmax']
            left_a = row['xmin']
            #top, bottom, right, left
            #coordinates = (row['ymin'],row['ymax'],row['xmin'],row['xmax'])
            #print(coordinates)
            #line will atleast have the word in it
            #line1=[row['Object']]
            #if below == False: 
            for idx_2, row_2 in df.iterrows():

                #check for higher idx values 
                #if idx_2 not in [x for v in bottom_connections.values() for x in v] and idx < idx_2:
                if idx_2 not in bottom_connections.values() and idx < idx_2:
                    #if idx_2 not in bottom_connections.values() and (idx != idx_2):
                        right_b = row_2['xmax']
                        left_b = row_2['xmin'] 
                        if (left_b <= right_a) and (right_b >= left_a): 
                            bottom_connections[idx] = idx_2
                            
                            top_connections[idx_2] = idx

                            #add it to the dataframe
                            df.loc[df['index'] == idx , 'bottom'] = idx_2
                            df.loc[df['index'] == idx_2, 'top'] = idx 

                            #print(bottom_connections)

                            #once the condition is met, break the loop to reduce redundant time complexity
                            break 
                    
                            #below = True 

    # print(df)


    # print(bottom_connections)
    # print(top_connections)
    # print(horizontal_connections)


    #combining both 
    result = {}
    dic1 = horizontal_connections
    dic2 = bottom_connections

    for key in (dic1.keys() | dic2.keys()):
        if key in dic1: result.setdefault(key, []).append(dic1[key])
        if key in dic2: result.setdefault(key, []).append(dic2[key])
    #print(result)


     
    G = nx.from_dict_of_lists(result)
    layout = nx.spring_layout(G)

    if show == True:
        nx.draw(G, layout, with_labels=True)
        plt.show()

    return G, df 

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

#features calculation

graph, processed_df = grapher(df)
print(df)
#nx.draw(graph, nx.spring_layout(graph), with_labels=True)
plt.show()

#locate df.index and see right, if right then do calculation with it and put it back 
#as a new column


def relative_distance(df):
    #RDL and RDT are negative while RDR and RDB are positive

    for index in df['index'].to_list():

        right_index = df.loc[df['index'] == index, 'right'].values[0]
        left_index = df.loc[df['index'] == index, 'left'].values[0]
        bottom_index = df.loc[df['index'] == index, 'bottom'].values[0]
        top_index = df.loc[df['index'] == index, 'top'].values[0]

        #rd_r = (right_word_xmin - left_word_xmax)/image_width

        #check if it is nan value 
        if np.isnan(right_index) == False: 
            right_word_left = df.loc[df['index'] == right_index, 'xmin'].values[0]
            source_word_right = df.loc[df['index'] == index, 'xmax'].values[0]

            df.loc[df['index'] == index, 'rd_r'] = (right_word_left - source_word_right)/image_width

        if np.isnan(left_index) == False:
            left_word_right = df.loc[df['index'] == left_index, 'xmax'].values[0]
            source_word_left = df.loc[df['index'] == index, 'xmin'].values[0]

            df.loc[df['index'] == index, 'rd_l'] = (left_word_right - source_word_left)/image_width
        
        if np.isnan(bottom_index) == False:
            bottom_word_top = df.loc[df['index'] == bottom_index, 'ymin'].values[0]
            source_word_bottom = df.loc[df['index'] == index, 'ymax'].values[0]

            df.loc[df['index'] == index, 'rd_b'] = (bottom_word_top - source_word_bottom)/image_height

        if np.isnan(top_index) == False:
            top_word_bottom = df.loc[df['index'] == top_index, 'ymax'].values[0]
            source_word_top = df.loc[df['index'] == index, 'ymin'].values[0]

            df.loc[df['index'] == index, 'rd_t'] = (top_word_bottom - source_word_top)/image_height


    #replace all tne NaN values with '0' meaning there is nothing in that direction
    df[['rd_r','rd_b','rd_l','rd_t']] = df[['rd_r','rd_b','rd_l','rd_t']].fillna(0)
    
    print(df)



relative_distance(processed_df)