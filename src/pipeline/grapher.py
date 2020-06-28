import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt 
import math 
import itertools
# for making adjacency matrix
import networkx as nx




df = pd.read_csv("000.csv")
image = cv2.imread("000.jpg")



image = cv2.imread("../../data/raw/img/339.jpg")


filename = '339.csv'
filename = '000.csv'

image = cv2.imread("../../data/raw/img/000.jpg")

filepath = '../../data/raw/box/'+filename
df = pd.read_csv(filepath, header=None, sep='\n')
df = df[0].str.split(',', expand=True)
temp = df.copy() 
temp[temp.columns] = temp.apply(lambda x: x.str.strip())
temp.fillna('', inplace=True)
temp[8]= temp[8].str.cat(temp.iloc[:,9:-1], sep =", ") 
temp[temp.columns] = temp.apply(lambda x: x.str.rstrip(", ,"))
temp = temp.loc[:, :8]
temp.drop([2,3,6,7], axis=1, inplace=True)
temp.columns = ['xmin','ymin','xmax','ymax','Object']
temp[['xmin','ymin','xmax','ymax']] = temp[['xmin','ymin','xmax','ymax']].apply(pd.to_numeric)

df = temp 
print(df)

#print(df)
#for relative distances later
image_height, image_width = image.shape[0], image.shape[1]

# df = pd.read_csv("../../data/raw/box/test_tess_339.csv")
# image = cv2.imread('../../data/raw/img/test_tess_339.jpg')
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
    #remove empty spaces both in front and behind
    df.columns = df.columns.str.strip()

    #further cleaning
    df.dropna(inplace=True)
    #sort from top to bottom
    df.sort_values(by=['ymin'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(df)

    #subtracting ymax by 1 to eliminate ambiguity of boxes being in both left and right 
    df["ymax"] = df["ymax"].apply(lambda x: x - 1)

    #print(df)
    #print(df)
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
def grapher(df, export_graph =False):

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
    
 
    
    #verticle connections formation


    bottom_connections = {}
    top_connections = {}

    for idx, row in df.iterrows():
        if idx not in bottom_connections.keys():
  
            right_a = row['xmax']
            left_a = row['xmin']

            for idx_2, row_2 in df.iterrows():

                #check for higher idx values 
             
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


    # print('bottom connections:', bottom_connections)
    # # print(top_connections)
    # print('horizontal connections:', horizontal_connections)


    #combining both 
    result = {}
    dic1 = horizontal_connections
    dic2 = bottom_connections

    for key in (dic1.keys() | dic2.keys()):
        if key in dic1: result.setdefault(key, []).append(dic1[key])
        if key in dic2: result.setdefault(key, []).append(dic2[key])
    #print(result)


     
    G = nx.from_dict_of_lists(result)
  

    if export_graph:
        file, _ = os.path.splitext(filename)
        plot_path ='../../figures/' + file + 'plain_graph' '.jpg'
        if not os.path.exists(plot_path):
            layout = nx.kamada_kawai_layout(G)        
            nx.draw(G, layout, with_labels=True)
            plt.savefig(plot_path, format="PNG")
            plt.show()

    return result, G, df 



#features calculation

dict_graph, graph, processed_df = grapher(df, export_graph=True) #, show=True)
# print(df)
# layout = nx.kamada_kawai_layout(graph)
# #layout = nx.spring_layout(graph)

# nx.draw(graph, layout, with_labels=True)
# plt.show()

#locate df.index and see right, if right then do calculation with it and put it back 
#as a new column


def relative_distance(df):
    #RDL and RDT are negative while RDR and RDB are positive
    plot_df = df.copy() 


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

            """
            for plotting purposes
            getting the mid point of the values to draw the lines for the graph
            mid points of source and destination for the bounding boxes
            """

            right_word_x_max = df.loc[df['index'] == right_index, 'xmax'].values[0]
            right_word_y_max = df.loc[df['index'] == right_index, 'ymax'].values[0]
            right_word_y_min = df.loc[df['index'] == right_index, 'ymin'].values[0]

            #source_word_x_min = df.loc[df['index'] == index, 'xmin'].values[0]

            #source_word_y_min = df.loc[df['index'] == index, 'xmin'].values[0]
            #source_word_y_min = df.loc[df['index'] == index, 'xmin'].values[0]



            plot_df.loc[df['index'] == index, 'destination_x_hori'] = (right_word_x_max + right_word_left)/2
            plot_df.loc[df['index'] == index, 'destination_y_hori'] = (right_word_y_max + right_word_y_min)/2
            #plot_df.loc[df['index'] == index, 'source_x_hori'] = (source_word_right - source_word_x_min)/2
            #plot_df.loc[df['index'] == index, 'source_y_hori'] = (source_word_right - source_word_min)/2



        if np.isnan(left_index) == False:
            left_word_right = df.loc[df['index'] == left_index, 'xmax'].values[0]
            source_word_left = df.loc[df['index'] == index, 'xmin'].values[0]

            df.loc[df['index'] == index, 'rd_l'] = (left_word_right - source_word_left)/image_width
        


        if np.isnan(bottom_index) == False:
            bottom_word_top = df.loc[df['index'] == bottom_index, 'ymin'].values[0]
            source_word_bottom = df.loc[df['index'] == index, 'ymax'].values[0]

            df.loc[df['index'] == index, 'rd_b'] = (bottom_word_top - source_word_bottom)/image_height


            #for plotting purposes
            bottom_word_top_max = df.loc[df['index'] == bottom_index, 'ymax'].values[0]
            #source_word_min = df.loc[df['index'] == index, 'ymin'].values[0]
            bottom_word_x_max = df.loc[df['index'] == bottom_index, 'xmax'].values[0]
            bottom_word_x_min = df.loc[df['index'] == bottom_index, 'xmin'].values[0]


            plot_df.loc[df['index'] == index, 'destination_y_vert'] = (bottom_word_top_max + bottom_word_top)/2
            plot_df.loc[df['index'] == index, 'destination_x_vert'] = (bottom_word_x_max + bottom_word_x_min)/2
            #plot_df.loc[df['index'] == index, 'source_y_vert'] = (source_word_bottom - source_word_min)/2


        if np.isnan(top_index) == False:
            top_word_bottom = df.loc[df['index'] == top_index, 'ymax'].values[0]
            source_word_top = df.loc[df['index'] == index, 'ymin'].values[0]

            df.loc[df['index'] == index, 'rd_t'] = (top_word_bottom - source_word_top)/image_height


    #replace all tne NaN values with '0' meaning there is nothing in that direction
    df[['rd_r','rd_b','rd_l','rd_t']] = df[['rd_r','rd_b','rd_l','rd_t']].fillna(0)
    

    plot_df['rd_r'] = df['rd_r']
    plot_df['rd_b'] = df['rd_b']
  
    return df, plot_df 





df, plot_df = relative_distance(processed_df)
print(plot_df)




def show_document_graph(plot_df, img, export_image=False):
    for idx, row in plot_df.iterrows():
        #bounding box
        cv2.rectangle(img, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (0, 0, 255), 2)

        if np.isnan(row['destination_x_vert']) == False:
            source_x = (row['xmax'] + row['xmin'])/2
            source_y = (row['ymax'] + row['ymin'])/2
            

            cv2.line(img, 
                    (int(source_x), int(source_y)),
                    (int(row['destination_x_vert']), int(row['destination_y_vert'])), 
                    (0,255,0), 1)


            text = "{:.3f}".format(row['rd_b'])
            text_coordinates = ( int((row['destination_x_vert'] + source_x)/2) , int((row['destination_y_vert'] +source_y)/2))     
            cv2.putText(img, text, text_coordinates, cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,0,0), 1)

            #text_coordinates = ((row['destination_x_vert'] + source_x)/2 , (row['destination_y_vert'] +source_y)/2)
        
        if np.isnan(row['destination_x_hori']) == False:
            source_x = (row['xmax'] + row['xmin'])/2
            source_y = (row['ymax'] + row['ymin'])/2

            cv2.line(img, 
                (int(source_x), int(source_y)),
                (int(row['destination_x_hori']), int(row['destination_y_hori'])), \
                (0,255,0), 1)

            text = "{:.3f}".format(row['rd_r'])
            text_coordinates = (int((row['destination_x_hori'] + source_x)/2) , int((row['destination_y_hori'] +source_y)/2))     
            cv2.putText(img, text, text_coordinates, cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,0,0), 1)

        


    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if export_image:
        file, _ = os.path.splitext(filename)
        plot_path ='../../figures/' + file + 'docu_graph' '.jpg'
        if not os.path.exists(plot_path):
            cv2.imwrite(plot_path, img)
  

  

print(dict_graph)

show_document_graph(plot_df,image, export_image=True)





exit() 
# import flair 
# from flair.data import Sentence
# from flair.embeddings import WordEmbeddings
# from flair.embeddings import CharacterEmbeddings
# from flair.embeddings import StackedEmbeddings
# from flair.embeddings import FlairEmbeddings
# from flair.embeddings import BertEmbeddings
# from flair.embeddings import ELMoEmbeddings
# from flair.embeddings import FlairEmbeddings


# flair_forward  = FlairEmbeddings('news-forward-fast')
# flair_backward = FlairEmbeddings('news-backward-fast')


# stacked_embeddings = StackedEmbeddings( embeddings = [ 
#                                                        flair_forward, 
#                                                        flair_backward
#                                                       ])


# # create a sentence #
# sentence = Sentence('Analytics Vidhya blogs are Awesome')
# # embed words in sentence #
# stacked_embeddings(sentence)
# for token in sentence:
#   print(token.embedding)
# # data type and size of embedding #
# print(type(token.embedding))
# # storing size (length) #
# z = token.embedding.size()[0]


# from tqdm import tqdm ## tracks progress of loop ##

# # creating a tensor for storing sentence embeddings #
# s = torch.zeros(0,z)

# # iterating Sentence (tqdm tracks progress) #
# for tweet in tqdm(txt):   
#   # empty tensor for words #
#   w = torch.zeros(0,z)   
#   sentence = Sentence(tweet)
#   stacked_embeddings.embed(sentence)
#   # for every word #
#   for token in sentence:
#     # storing word Embeddings of each word in a sentence #
#     w = torch.cat((w,token.embedding.view(-1,z)),0)
#   # storing sentence Embeddings (mean of embeddings of all words)   #
#   s = torch.cat((s, w.mean(dim = 0).view(-1, z)),0)


print(dict_graph)


def make_adjacency(graph_dict):#, text_list):
    '''
        Function to make an adjacency matrix from a networkx graph object
        as well as padded feature matrix
        Args:
            G: networkx graph object
            
            text_list: list,
                        of text entities:
                        ['Tax Invoice', '1/2/2019', ...]
        Returns:
            A: Adjacency matrix as np.array
            X: Feature matrix as numpy array for input graph
    '''
    G = nx.from_dict_of_lists(graph_dict)
    adj_sparse = nx.adjacency_matrix(G)

    # preprocess the sparse adjacency matrix returned by networkx function
    A = np.array(adj_sparse.todense())
    #A = self._pad_adj(A)

    # preprocess the list of text entities
    #feat_list = list(map(self._get_text_features, text_list))
    #feat_arr = np.array(feat_list)
    #X = self._pad_text_features(feat_arr)

    return A

adjacancy_matrix = make_adjacency(dict_graph)

print(adjacancy_matrix)
print(adjacancy_matrix.shape)

"""
def get_text_features(df): 
    data = df['Object'].tolist()
    
    '''
        Args:
            str, input data
            
        Returns: 
            np.array, shape=(22,);
            an array of the text converted to features
            
    '''
    special_chars = ['&', '@', '#', '(',')','-','+', 
                '=', '*', '%', '.', ',', '\\','/', 
                '|', ':']

    # character wise
    n_lower, n_upper, n_spaces, n_alpha, n_numeric,n_special = [],[],[],[],[],[]

    for words in data:
        upper,lower,alpha,spaces,numeric,special = 0,0,0,0,0,0
        for char in words: 
            print(char)
            # for lower letters 
            if char.islower(): 
                lower += 1
    
            # for upper letters 
            if char.isupper(): 
                upper += 1
            
            # for white spaces
            if char.isspace():
                spaces += 1
            
            # for alphabetic chars
            if char.isalpha():
                alpha += 1
            
            # for numeric chars
            if char.isnumeric():
                numeric += 1
                           
            if char in special_chars:
                special += 1 

        n_lower.append(lower)
        n_upper.append(upper)
        n_spaces.append(spaces)
        n_alpha.append(alpha)
        n_numeric.append(numeric)
        n_special.append(special)
        #features.append([n_lower, n_upper, n_spaces, n_alpha, n_numeric, n_digits])

    df['n_upper'],df['n_lower'],df['n_alpha'],df['n_spaces'],\
    df['n_numeric'],df['n_special'] = n_upper, n_lower, n_alpha, n_spaces, n_numeric,n_special

    print(df)
    print(df.loc[df['index'] == 75].Object)

get_text_features(df)

"""