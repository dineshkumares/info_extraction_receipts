


class Graph:
    
    def __init__(self):
        pass

    
    def _pad_adj(self, adj):
        """
			This method resizes the input Adjacency matrix to shape 
			(self.max_nodes, self.max_nodes)
			adj: 
				2d numpy array
				adjacency matrix
		"""
		
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

    def make_adjacency(self, graph_dict):#, text_list):
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

    
    def get_text_features(self, df): 
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

        
