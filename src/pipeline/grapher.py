import numpy as np
import pandas as pd
import cv2
import os


# for making adjacency matrix
import networkx as nx

def get_clean_df(filename):
    filepath = '../../data/raw/box/'+filename
    to_write_filepath = '../../data/interim/'+filename
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
    return temp



def connect(df, img, plot=False, export_df=False):

    
    '''
        This method implements the logic to generate a graph based on
        visibility. If a horizontal/vertical line can be drawn from one
        node to another, the two nodes are connected.
        Args:
            plot (default=False):
                bool, whether to plot the graph;
                the graph is plotted in at path ./grapher_outputs/plots
            export_df (default=False):
                bool, whether to export the dataframe containing graph 
                information;
                the dataframe is exported as csv to path 
                ./grapher_outputs/connections
    '''
    # initialize empty df to store plotting coordinates
    df_plot = pd.DataFrame()
    count = 0
    img = cv2.imread(img)

    # initialize empty lists to store coordinates and distances
    # ================== vertical======================================== #
    distances, nearest_dest_ids_vert = [], []
    
    x_src_coords_vert, y_src_coords_vert, x_dest_coords_vert, \
    y_dest_coords_vert = [], [], [], []
    
    # ======================= horizontal ================================ #
    lengths, nearest_dest_ids_hori = [], []
    
    x_src_coords_hori, y_src_coords_hori, x_dest_coords_hori, \
    y_dest_coords_hori = [], [], [], []

    for src_idx, src_row in df.iterrows():
        
        # ================= vertical ======================= #
        src_range_x = (src_row['xmin'], src_row['xmax'])
        src_center_y = (src_row['ymin'] + src_row['ymax'])/2

        dest_attr_vert = []

        # ================= horizontal ===================== #
        src_range_y = (src_row['ymin'], src_row['ymax'])
        src_center_x = (src_row['xmin'] + src_row['xmax'])/2
        
        dest_attr_hori = []

        ################ iterate over destination objects #################
        for dest_idx, dest_row in df.iterrows():
            # flag to signal whether the destination object is below source
            is_beneath = False
            if not src_idx == dest_idx:
                # ==================== vertical ==========================#
                dest_range_x = (dest_row['xmin'], dest_row['xmax'])
                dest_center_y = (dest_row['ymin'] + dest_row['ymax'])/2
                
                height = dest_center_y - src_center_y

                # consider only the cases where destination object lies 
                # below source
                if dest_center_y > src_center_y:
                    # check if horizontal range of dest lies within range 
                    # of source

                    # case 1
                    if dest_range_x[0] <= src_range_x[0] and \
                        dest_range_x[1] >= src_range_x[1]:
                        
                        x_common = (src_range_x[0] + src_range_x[1])/2
                        
                        line_src = (x_common , src_center_y)
                        line_dest = (x_common, dest_center_y)

                        attributes = (dest_idx, line_src, line_dest, height)
                        dest_attr_vert.append(attributes)
                        
                        is_beneath = True

                    # case 2
                    elif dest_range_x[0] >= src_range_x[0] and \
                        dest_range_x[1] <= src_range_x[1]:
                        
                        x_common = (dest_range_x[0] + dest_range_x[1])/2
                        
                        line_src = (x_common, src_center_y)
                        line_dest = (x_common, dest_center_y)
                        
                        attributes = (dest_idx, line_src, line_dest, height)
                        dest_attr_vert.append(attributes)
                        
                        is_beneath = True

                    # case 3
                    elif dest_range_x[0] <= src_range_x[0] and \
                        dest_range_x[1] >= src_range_x[0] and \
                            dest_range_x[1] < src_range_x[1]:

                        x_common = (src_range_x[0] + dest_range_x[1])/2

                        line_src = (x_common , src_center_y)
                        line_dest = (x_common, dest_center_y)

                        attributes = (dest_idx, line_src, line_dest, height)
                        dest_attr_vert.append(attributes)

                        is_beneath = True

                    # case 4
                    elif dest_range_x[0] <= src_range_x[1] and \
                        dest_range_x[1] >= src_range_x[1] and \
                            dest_range_x[0] > src_range_x[0]:
                        
                        x_common = (dest_range_x[0] + src_range_x[1])/2
                        
                        line_src = (x_common , src_center_y)
                        line_dest = (x_common, dest_center_y)

                        attributes = (dest_idx, line_src, line_dest, height)
                        dest_attr_vert.append(attributes)

                        is_beneath = True
        
            if not is_beneath:
                # ======================= horizontal ==================== #
                    dest_range_y = (dest_row['ymin'], dest_row['ymax'])
                    # get center of destination NOTE: not used
                    dest_center_x = (dest_row['xmin'] + dest_row['xmax'])/2
                    
                    # get length from destination center to source center
                    if dest_center_x > src_center_x:
                        length = dest_center_x - src_center_x
                    else:
                        length = 0
                    
                    # consider only the cases where the destination object 
                    # lies to the right of source
                    if dest_center_x > src_center_x:
                        #check if vertical range of dest lies within range 
                        # of source
                        
                        # case 1
                        if dest_range_y[0] >= src_range_y[0] and \
                            dest_range_y[1] <= src_range_y[1]:
                            
                            y_common = (dest_range_y[0] + dest_range_y[1])/2

                            line_src = (src_center_x, y_common)
                            line_dest = (dest_center_x, y_common)
                            
                            attributes = (dest_idx, line_src, line_dest, length)
                            dest_attr_hori.append(attributes)

                        # case 2
                        if dest_range_y[0] <= src_range_y[0] and \
                            dest_range_y[1] <= src_range_y[1] and \
                                dest_range_y[1] > src_range_y[0]:
                            
                            y_common = (src_range_y[0] + dest_range_y[1])/2

                            line_src = (src_center_x, y_common)
                            line_dest = (dest_center_x, y_common)

                            attributes = (dest_idx, line_src, line_dest, length)
                            dest_attr_hori.append(attributes)

                        # case 3
                        if dest_range_y[0] >= src_range_y[0] and \
                            dest_range_y[1] >= src_range_y[1] and \
                                dest_range_y[0] < src_range_y[1]:

                            y_common = (dest_range_y[0] + src_range_y[1])/2

                            line_src = (src_center_x, y_common)
                            line_dest = (dest_center_x, y_common)

                            attributes = (dest_idx, line_src, line_dest, length)
                            dest_attr_hori.append(attributes)

                        # case 4
                        if dest_range_y[0] <= src_range_y[0] \
                            and dest_range_y[1] >= src_range_y[1]:

                            y_common = (src_range_y[0] + src_range_y[1])/2

                            line_src = (src_center_x, y_common)
                            line_dest = (dest_center_x, y_common)

                            attributes = (dest_idx, line_src, line_dest, length)
                            dest_attr_hori.append(attributes)

        # sort list of destination attributes by height/length at position 
        # 3 in tuple
        dest_attr_vert_sorted = sorted(dest_attr_vert, key = lambda x: x[3])
        dest_attr_hori_sorted = sorted(dest_attr_hori, key = lambda x: x[3])
        
        # append the index and source and destination coords to draw line 
        # ==================== vertical ================================= #
        if len(dest_attr_vert_sorted) == 0:
            nearest_dest_ids_vert.append(-1)
            x_src_coords_vert.append(-1)
            y_src_coords_vert.append(-1)
            x_dest_coords_vert.append(-1)
            y_dest_coords_vert.append(-1)
            distances.append(0)
        else:
            nearest_dest_ids_vert.append(dest_attr_vert_sorted[0][0])
            x_src_coords_vert.append(dest_attr_vert_sorted[0][1][0])
            y_src_coords_vert.append(dest_attr_vert_sorted[0][1][1])
            x_dest_coords_vert.append(dest_attr_vert_sorted[0][2][0])
            y_dest_coords_vert.append(dest_attr_vert_sorted[0][2][1])
            distances.append(dest_attr_vert_sorted[0][3])

        # ========================== horizontal ========================= #
        if len(dest_attr_hori_sorted) == 0:
            nearest_dest_ids_hori.append(-1)
            x_src_coords_hori.append(-1)
            y_src_coords_hori.append(-1)
            x_dest_coords_hori.append(-1)
            y_dest_coords_hori.append(-1)
            lengths.append(0)			
        
        else:
        # try and except for the cases where there are vertical connections
        # still to be made but all horizontal connections are accounted for
            try:
                nearest_dest_ids_hori.append(dest_attr_hori_sorted[0][0])
            except:
                nearest_dest_ids_hori.append(-1)
            
            try:
                x_src_coords_hori.append(dest_attr_hori_sorted[0][1][0])
            except:
                x_src_coords_hori.append(-1)

            try:
                y_src_coords_hori.append(dest_attr_hori_sorted[0][1][1])
            except:
                y_src_coords_hori.append(-1)

            try:
                x_dest_coords_hori.append(dest_attr_hori_sorted[0][2][0])
            except:
                x_dest_coords_hori.append(-1)

            try:
                y_dest_coords_hori.append(dest_attr_hori_sorted[0][2][1])
            except:
                y_dest_coords_hori.append(-1)

            try:
                lengths.append(dest_attr_hori_sorted[0][3])
            except:
                lengths.append(0)			

    # ==================== vertical ===================================== #
    # create df for plotting lines
    df['below_object'] = df.loc[nearest_dest_ids_vert, 'Object'].values

    # add distances column
    df['below_dist'] = distances
    
    # add column containing index of destination object
    df['below_obj_index'] = nearest_dest_ids_vert

    # add coordinates for plotting
    df_plot['x_src_vert'] = x_src_coords_vert
    df_plot['y_src_vert'] = y_src_coords_vert
    df_plot['x_dest_vert'] = x_dest_coords_vert
    df_plot['y_dest_vert'] = y_dest_coords_vert
    
    # df.fillna('NULL', inplace = True)

    # ==================== horizontal =================================== #
    # create df for plotting lines
    df['side_object'] = df.loc[nearest_dest_ids_hori, 'Object'].values

    # add lengths column
    df['side_length'] = lengths

    # add column containing index of destination object
    df['side_obj_index'] = nearest_dest_ids_hori

    # add coordinates for plotting
    df_plot['x_src_hori'] = x_src_coords_hori
    df_plot['y_src_hori'] = y_src_coords_hori
    df_plot['x_dest_hori'] = x_dest_coords_hori
    df_plot['y_dest_hori'] = y_dest_coords_hori

    ########################## concat df and df_plot ######################
    
    df_merged = pd.concat([df, df_plot], axis=1)
    
    # if an object has more than one parent above it, only the connection 
    # with the smallest distance is retained and the other distances are 
    # replaced by '-1' to get such objects, group by 'below_object' column 
    # and use minimum of 'below_dist'
    
    # ======================= vertical ================================== #
    groups_vert = df_merged.groupby('below_obj_index')['below_dist'].min()
    # groups.index gives a list of the below_object text and groups.values 
    # gives the corresponding minimum distance
    groups_dict_vert = dict(zip(groups_vert.index, groups_vert.values))

    # ======================= horizontal ================================ #
    groups_hori = df_merged.groupby('side_obj_index')['side_length'].min()
    # groups.index gives a list of the below_object text and groups.values 
    # gives the corresponding minimum distance
    groups_dict_hori = dict(zip(groups_hori.index, groups_hori.values))
    
        
    revised_distances_vert = []
    revised_distances_hori = []

    rev_x_src_vert, rev_y_src_vert, rev_x_dest_vert, rev_y_dest_vert = \
                                                            [], [], [], []
    rev_x_src_hori, rev_y_src_hori, rev_x_dest_hori, rev_y_dest_hori = \
                                                            [], [], [], []
    
    for idx, row in df_merged.iterrows():
        below_idx = row['below_obj_index']
        side_idx = row['side_obj_index']

        # ======================== vertical ============================= #
        if row['below_dist'] > groups_dict_vert[below_idx]:
            revised_distances_vert.append(-1)
            rev_x_src_vert.append(-1)
            rev_y_src_vert.append(-1)
            rev_x_dest_vert.append(-1)
            rev_y_dest_vert.append(-1)

        else:
            revised_distances_vert.append(row['below_dist'])
            rev_x_src_vert.append(row['x_src_vert'])
            rev_y_src_vert.append(row['y_src_vert'])
            rev_x_dest_vert.append(row['x_dest_vert'])
            rev_y_dest_vert.append(row['y_dest_vert'])

        # ========================== horizontal ========================= #
        if row['side_length'] > groups_dict_hori[side_idx]:
            revised_distances_hori.append(-1)
            rev_x_src_hori.append(-1)
            rev_y_src_hori.append(-1)
            rev_x_dest_hori.append(-1)
            rev_y_dest_hori.append(-1)

        else:
            revised_distances_hori.append(row['side_length'])
            rev_x_src_hori.append(row['x_src_hori'])
            rev_y_src_hori.append(row['y_src_hori'])
            rev_x_dest_hori.append(row['x_dest_hori'])
            rev_y_dest_hori.append(row['y_dest_hori'])

    # store in dataframe
    # ============================ vertical ============================= #
    df['revised_distances_vert'] = revised_distances_vert
    df_merged['x_src_vert'] = rev_x_src_vert
    df_merged['y_src_vert'] = rev_y_src_vert
    df_merged['x_dest_vert'] = rev_x_dest_vert
    df_merged['y_dest_vert'] = rev_y_dest_vert

    # ======================== horizontal =============================== #
    df['revised_distances_hori'] = revised_distances_hori
    df_merged['x_src_hori'] = rev_x_src_hori
    df_merged['y_src_hori'] = rev_y_src_hori
    df_merged['x_dest_hori'] = rev_x_dest_hori
    df_merged['y_dest_hori'] = rev_y_dest_hori
    
    
    # plot image if plot==True 
    if plot == True:
    
        # make folder to store output
        if not os.path.exists('grapher_outputs'):
            os.makedirs('grapher_outputs')	
        
        # subdirectory to store plots
        if not os.path.exists('./grapher_outputs/plots'):
            os.makedirs('./grapher_outputs/plots')
        
        # check if image exists in folder
        try: 
            if len(img) == None:
                pass
        except: 
            pass
        
        # plot if image exists
        else:
            for idx, row in df_merged.iterrows():
                cv2.line(img, 
                        (int(row['x_src_vert']), int(row['y_src_vert'])), 
                        (int(row['x_dest_vert']), int(row['y_dest_vert'])), 
                        (0,0,255), 2)
                
                cv2.line(img, 
                        (int(row['x_src_hori']), int(row['y_src_hori'])), 
                        (int(row['x_dest_hori']), int(row['y_dest_hori'])), 
                        (0,0,255), 2)

            # write image in same folder
            PLOT_PATH = \
                './grapher_outputs/plots/' + 'object_tree_' + str(count) + '.jpg'
            cv2.imwrite(PLOT_PATH, img)


    # export dataframe with destination objects to csv in same folder
    if export_df == True:
        
        # make folder to store output
        if not os.path.exists('grapher_outputs'):
            os.makedirs('grapher_outputs')			
        
        # subdirectory to store plots
        if not os.path.exists('./grapher_outputs/connections'):
            os.makedirs('./grapher_outputs/connections')

        CSV_PATH = \
            './grapher_outputs/connections/' + 'connections_' + str(count) + '.csv'
        df.to_csv(CSV_PATH, index = None)
            
    # convert dataframe to dict:
    # {src_id: dest_1, dest_2, ..}

    graph_dict = {}
    for src_id, row in df.iterrows():
        if row['below_obj_index'] != -1:
            graph_dict[src_id] = [row['below_obj_index']]
        if row['side_obj_index'] != -1:
            graph_dict[src_id] = [row['side_obj_index']]
        
    
    return graph_dict, df['Object'].tolist()



def _pad_adj(adj):
    '''
        This method resizes the input Adjacency matrix to shape 
        (self.max_nodes, self.max_nodes)
        adj: 
            2d numpy array
            adjacency matrix
    '''
    
    assert adj.shape[0] == adj.shape[1], f'The input adjacency matrix is \
        not square and has shape {adj.shape}'
    
    # get n of nxn matrix
    n = adj.shape[0]
    
    if n < self.max_nodes:
        target = np.zeros(shape=(self.max_nodes, self.max_nodes))

        # fill in the target matrix with the adjacency
        target[:adj.shape[0], :adj.shape[1]] = adj
        
    elif n > self.max_nodes:
        # cut away the excess rows and columns of adj
        target = adj[:self.max_nodes, :self.max_nodes]
        
    else:
        # do nothing
        target = adj
        
    return target


def make_adjacency(graph_dict, text_list):
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
		#A = _pad_adj(A)

		return A



if __name__ == '__main__':
    df = get_clean_df('000.csv')
    image = "../../data/raw/img/000.jpg"
    #connect(df, image,plot=True,export_df=True)
    graph, object_list = connect(df, image)
    print(graph, object_list)
    A = make_adjacency(graph,object_list)
    print(A)
    print(A.shape)
    nx.draw(G, layout, with_labels=True)