# # """



# from bpemb import BPEmb
# bpemb_en = BPEmb(lang="ms",dim=100)

# print(bpemb_en.encode("Gerbang"))

# print(type(bpemb_en.vectors))


# print(sum(bpemb_en.vectors))

# print(bpemb_en.vectors.shape)



# bpemb_en = BPEmb(lang="en", dim=100)

# print(bpemb_en.encode("Gerbang"))

# print(type(bpemb_en.vectors))

# print(bpemb_en.vectors.shape)

# print(sum(bpemb_en.vectors))

# from flair.embeddings import BytePairEmbeddings
# from flair.data import Sentence

# embedding = BytePairEmbeddings('en')



# sentence = Sentence('The grass is green .')

# sentence = Sentence('10')
# sentence = Sentence('ten')

# embedding.embed(sentence)


# for token in sentence:
#     print(token)
#     print(token.embedding)
#     print(token.embedding.shape)
    
# import pandas as pd 

# filename = '000'
# file_path = "../../data/raw/box/" + filename + '.csv'
# df = pd.read_csv(file_path, header=None, sep='\n')

# df = df[0].str.split(',', expand=True)
# temp = df.copy() 
# temp[temp.columns] = temp.apply(lambda x: x.str.strip())
# temp.fillna('', inplace=True)
# temp[8]= temp[8].str.cat(temp.iloc[:,9:], sep =", ") 
# print(df)
# temp[temp.columns] = temp.apply(lambda x: x.str.rstrip(", ,"))
# temp = temp.loc[:, :8]
# temp.drop([2,3,6,7], axis=1, inplace=True)
# temp.columns = ['xmin','ymin','xmax','ymax','Object']
# temp[['xmin','ymin','xmax','ymax']] = temp[['xmin','ymin','xmax','ymax']].apply(pd.to_numeric)
# print(temp)

# import pandas as pd 
# import glob 
# import os 

# path = '/Users/udipbohara/Desktop/Datascience_projects/info_extraction_receipts/data/interim/'
# # path2 = '/Users/udipbohara/Desktop/interim/'
# # final_path = '/Users/udipbohara/Desktop/cleaned_dfs/'


# # files = []
# # first_list = []
# # second_list = []


# for file in glob.glob(os.path.join(path, '*.csv')):
   

#     df1 =  pd.read_csv(file)

#     if len(df1.columns) != 10:
    
#         raise ValueError('You done goofed in {}'.format(file) )
#     else:
#         pass
# import torch
# from torch.nn import Parameter

# in_channels = 12
# out_channels = 15
# weight = Parameter(torch.Tensor(in_channels, out_channels))
import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

print(data.y)
print(data.y.unique())
print(data.__dict__)