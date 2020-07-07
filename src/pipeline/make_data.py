from graph import Grapher
import torch
import scipy.sparse
import torch_geometric.data
import networkx as nx
import numpy as np
import os 



def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def from_networkx(G):
    """Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    G = nx.convert_node_labels_to_integers(G)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass 

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
    return data





def get_data():
    path = "../../data/raw/box/"
    l=os.listdir(path)
    files=[x.split('.')[0] for x in l]
    files.sort()
    files = files[1:]

    list_of_graphs = []

    for file in files[:30]:

        connect = Grapher(file)
        G,_,_ = connect.graph_formation()
        df = connect.relative_distance()
        individual_data = from_networkx(G)

        feature_cols = ['xmin', 'ymin', 'xmax', 'ymax','rd_b','line_number', 'rd_r', 'rd_t', 'rd_l']
        features = torch.tensor(df[feature_cols].values.astype(np.float32))

        df['labels'] = df['labels'].fillna('undefined')
        df.loc[df['labels'] == 'company', 'num_labels'] = 1
        df.loc[df['labels'] == 'address', 'num_labels'] = 2
        df.loc[df['labels'] == 'invoice', 'num_labels'] = 3
        df.loc[df['labels'] == 'date', 'num_labels'] = 4
        df.loc[df['labels'] == 'total', 'num_labels'] = 5
        df.loc[df['labels'] == 'undefined', 'num_labels'] = 6
 
        labels = torch.tensor(df['num_labels'].values.astype(np.int))

        individual_data.x = features
        individual_data.y = labels

        print(individual_data)

        list_of_graphs.append(individual_data)
    

    data = torch_geometric.data.Batch.from_data_list(list_of_graphs)
    

    data.edge_attr = None 

    #50,20,30 split for train,val,split
    length_labels = len(data.y)
    x = [_ for _ in range(length_labels)]
    np.random.shuffle(x)
    training_num = int(50/100 * length_labels) 
    validation_num  = int(20/100 * length_labels )
    testing_num = length_labels - (training_num + validation_num)

    idx_train, idx_val, idx_test=\
        x[:training_num], x[training_num:validation_num+training_num],\
        x[-testing_num:]


    idx_train, idx_val, idx_test = torch.LongTensor(idx_train),\
                                torch.LongTensor(idx_val),\
                                torch.LongTensor(idx_test)

    data.train_mask = sample_mask(idx_train, data.y.shape[0])
    data.val_mask = sample_mask(idx_val, data.y.shape[0])
    data.test_mask = sample_mask(idx_test, data.y.shape[0])

    save_path = "../../data/processed/"  
    torch.save(data, save_path +'data.dataset')



if __name__ == "__main__":
    #get_data()
    data_path = "../../data/processed/" + "data.dataset"
    data = torch.load(data_path)
    print(data)
    print(data.__dict__)
    print(data.train_mask)
    