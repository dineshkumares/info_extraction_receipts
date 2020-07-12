import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt 

df = pd.DataFrame({'DOC': ['Doc_A', 'Doc_B', 'Doc_C', 'Doc_D', 'Doc_E'], 'topic_A': [0,0,1,0,0], 'topic_B': [1,0,0,1,0], 'topic_C': [0,1,1,1,0]})
print(df)
df1 = df.set_index(['DOC']).stack().rename('Status').reset_index().query('Status != 0')

print(df1)
G = nx.from_pandas_edgelist(df1,'level_1','DOC')

D,T = nx.bipartite.sets(G)
pos = dict()
pos.update( (n, (1, i)) for i, n in enumerate(D) ) 
pos.update( (n, (2, i)) for i, n in enumerate(T) ) 
nx.draw(G, pos=pos, alpha=.4)


for i in pos:
    x, y = pos[i]
    plt.text(x-.05, y+.2, i)

plt.show() 