info_extraction_receipts
==============================
# Why Graphs?

"The aggregation over the neighborhood of a node in the network is analogue to a pooling operation in a convolutional neural network and the multiplication with the weight matrix W is analogue to a filtering operation. Although these operations are similar — hence the similarity in names between GCNs and CNNs — they are not the same.
"

## Classification:
_edit this_ <br>
When dealing with large graphs, such as those that arise in the context of online social networks, a subset of nodes may be labeled. These labels can indicate demographic values, interest, beliefs or other characteristics of the nodes (users). A core problem is to use this information to extend the labeling so that all nodes are assigned a label (or labels).

### ubiquitous
## Applying Graph Convolutional Networks:
(Simplifying Graph Convolutional Networks)
https://arxiv.org/pdf/1902.07153.pdf

# GCN FROM SCRATCH

https://colab.research.google.com/drive/1HGSmZYjmyXlVsqp9LvlIOXxjIRWW_jmz


# solution for classification:
https://colab.research.google.com/drive/1lpSXzmkBac4122aXpZgoShCnwEPZ7xgL#scrollTo=I2rM9GtBLLjF


# Build dataset from this?

Use Wikidata?

wikidata.org 

https://github.com/networkx/networkx/tree/master/examples/applications

Or

__Classification Datasets:__
https://medium.com/@sergei.ivanov_24894/new-graph-classification-data-sets-43e134340d2d



__More data:__
http://www-personal.umich.edu/~mejn/netdata/




# Data
http://www-personal.umich.edu/~mejn/netdata/


<!--  -->




### Classification using GCNs

Simplifying Graph Convlutional Networks
https://arxiv.org/pdf/1902.07153.pdf

Node features and attributes are represented by 'signals'

# Graph Convolutional Networks (GCNs)

Graph Neural Networks is a subset of deep learning with a lot of stuff done in it.Check this out! [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/pdf/1812.08434.pdf)


Check out [this](https://towardsdatascience.com/graph-convolutional-networks-for-geometric-deep-learning-1faf17dee008) excellent introduction to Graph Neural Networks.

OF the subset, here is another subset: 
## Spectral Convolutions
Blog post by the author:
https://tkipf.github.io/graph-convolutional-networks/
<br>
_Convolution_ 
: In mathematics convolution is a mathematical operation on two functions that produces a third function expressing how the shape of one is modified by the other.



Convolutions can be computed in by finding the eigendecomposition of the graph Laplacian

Why Conventional Convolutional Neural Network methods dont work:
- There is no Euclidean distance in graphs.
- Graphs have a certain number of nodes. 
- There is no notion of direction in graphs.

Overall steps are
- Transform the graph into the spectral domain using eigendecomposition
- Apply eigendecomposition to the specified kernel
- Multiply the spectral graph and spectral kernel (like vanilla convolutions)
- Return results in the original spatial domain (analogous to inverse GFT)

A kernel is a matrix, which is slid across the image and multiplied with the input such that the output is enhanced in a certain desirable manner.

# Give example of a graph and adjacent, degree and laplacian matrix

### Laplacian matrix ( & Eigen vectors &)
The Laplacian is used in calculating “graph gradients,” i.e. measures of how much the function values is changing at each node, relative to the function value at nearby nodes, according to the weight with each of those neighbor nodes. __ When you calculate the eigenvectors of the Laplacian, you end up with a Fourier basis for the graph,,This enables us to project the graph onto a N-dimensional Euclidean space and results in axes also known as '__spectrum__' of the graph,__ Hence the name __Spectral Convolution__

It’s possible to transform both your data and your filters such that each filter is represented by a simple diagonalized matrix (all zeroes, except for values along the diagonal), that gets multiplied against a transformed version of the data. This simplification happens when you perform a Fourier Transform of your data. Fourier transforms are typically thought of in the context of functions, and (on a very high level), they represent any function as a weighted composition of simpler “frequency” functions; starting with low frequencies that explain the broad strokes pattern of the data, and ending with high frequencies that fill in the smaller details.

Principle behind this:
- Calculate the Ajacent/weight matrix and Diagonal matrix for a graph
- Calculate the Laplacian matrix ( D- A) (give a sense of euclidean space)
- Calculate the eigenvectors of the Laplacian matrix (Spectral domain)
- You end up with a Fourier basis for the graph. 
 Fourier Transform [video](https://www.youtube.com/watch?v=spUNpyF58BY).
To build an intution of how it all ties to Fourier Transform, [this](https://www.math.ucla.edu/~tao/preprints/fourier.pdf) is an excellent paper. 
__Key take-away__: When you calculate the eigenvectors of the Laplacian, you end up with a Fourier basis for the graph

- Project both weights and data into that basis (results in a convolution)

To calculate spectral convolutions through this method, you first need to calculate the full NxN Laplacian eigenvector matrix of the graph, and then multiply each filter, as well as the data itself, by that matrix. This is quite computationally intensive for large graphs, since it scales by N² for each filter.

<br>
_It’s possible to transform both your data and your filters such that each filter is represented by a simple diagonalized matrix (all zeroes, except for values along the diagonal), that gets multiplied against a transformed version of the data. This simplification happens when you perform a Fourier Transform of your data_

To summarize it in code form:

# Main Paper: 
SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS 
https://arxiv.org/pdf/1609.02907.pdf

In Kipf and Welling’s Graph Convolutional Network, a convolution is defined by: 

![g_{\theta {}'} \star x \approx \sum_{k=0}^{K} \theta {}'_{k}T_{k}(\tilde{L})x](https://render.githubusercontent.com/render/math?math=g_%7B%5Ctheta%20%7B%7D'%7D%20%5Cstar%20x%20%5Capprox%20%5Csum_%7Bk%3D0%7D%5E%7BK%7D%20%5Ctheta%20%7B%7D'_%7Bk%7DT_%7Bk%7D(%5Ctilde%7BL%7D)x)


Where gθ is a kernel (θ represents the parameters) which is applied (represented by the star) to x, a graph signal. K stands for the number of nodes away from the target node to consider (the Kth order neighborhood, with k being the the closest order neighbor). T denotes the Chebyshev polynomials as applied to L̃ which represents the equation:

![\tilde{L} = \frac{2}{\lambda _{max}}L - I_{N}](https://render.githubusercontent.com/render/math?math=%5Ctilde%7BL%7D%20%3D%20%5Cfrac%7B2%7D%7B%5Clambda%20_%7Bmax%7D%7DL%20-%20I_%7BN%7D)


Where λ max denotes the largest eigenvalue of L, the normalized graph laplacian. Multiple convolutions can be performed on a graph, and the output is aggregated into Z.

__This is the main point :__ 

![Z = \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}X\Theta ](https://render.githubusercontent.com/render/math?math=Z%20%3D%20%5Ctilde%7BD%7D%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%5Ctilde%7BA%7D%5Ctilde%7BD%7D%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7DX%5CTheta%20)

Where Z is a matrix of convolved signals (from neighboring nodes) Ã is the adjacency matrix of the graph (plus the identity matrix), D̃ is the diagonal node degree from Ã, Θ is a matrix of kernel/filter parameters (can be shared over the whole graph), and X is a matrix of node feature vectors. The D to the power of -1/2 are parts of a renormalization trick to avoid both exploding or vanishing gradients.



In a general/broad sense, the Fourier transform is a systematic way to decompose “generic” functions into a superposition of “symmetric” functions.



"The Fourier transform is also related to topics in linear algebra, such as the representation of a vector as linear combinations of an orthonormal basis, or as linear combinations of eigenvectors of a matrix (or a linear operator)." - _Tao_

better explanation of that can be found [here](https://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
)



A graph can be represented as a matrix. For example:
-> 
<br> as you can see, then you can have an adjacent matrix.
Then you can have a diagonal matrix.

From that you can calculate the Laplacian matrix .

\[L = D - A\]
Insert latex in MD: 
https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b

<img src="https://render.githubusercontent.com/render/math?math=\int_{a}^{b} f(x)dx = F(b) - F(a)">




| Syntax | Description |
| ----------- | ----------- |
| Header | Title |
| Paragraph | Text |



Chebyshev polynomial
------------------------

The kernel equals the sum of all Chebyshev polynomial kernels applied to the diagonal matrix of scaled Laplacian eigenvalues for each order of k up to K-1.

First order simply means that the metric used to determine the similarity between 2 nodes is based on the node’s immediate neighborhood. Second order (and beyond) means the metric used to determine similarity considers a node’s immediate neighborhood, but also the similarities between the neighborhood structures (with each increasing order, the depth of which nodes are considered increases).

__Consider avoiding Chebyshev to avoid overfitting.__

__ChebNets and GCNs are very similar, but their largest difference is in their choices for value K in eqn. 1. In a GCN, the layer wise convolution is limited to K = 1__


Steps:
----------------------
From the paper: 
- Eigen decomposition of Laplacian (fourier Transform) gives the filter 
- use 4 GCN layers ( first layer learns 16 different filters/graph patterns(neurons))
- second learns 32
- third learns 64
- fourth is the output layer

The aggregation over the neighborhood of a node in the network is analogue to a pooling operation in a convolutional neural network and the multiplication with the weight matrix W is analogue to a filtering operation. Although these operations are similar — hence the similarity in names between GCNs and CNNs — they are not the same.






# For scalability:
https://arxiv.org/pdf/1902.07153.pdf





end-to-end information extraction from receipts using GCNs

To learn more about the dataset, you can read [here](http://www.cs.umd.edu/~sen/pubs/sna2006/RelClzPIT.pdf)

Inspired by this blog post:

[Information Extraction from Receipts with Graph Convolutional Networks](https://nanonets.com/blog/information-extraction-graph-convolutional-networks/?&utm_source=nanonets.com/blog/&utm_medium=blog&utm_content=Adrian%20Sarno%20-%20AI%20&%20Machine%20Learning%20Blog#commento-login-box-container)


Use this to visualize:

https://github.com/TobiasSkovgaardJepsen/gcn/blob/master/notebooks/3-tsj-community-prediction-in-zacharys-karate-club-with-gcn.ipynb


Read this:

Data?

https://github.com/zzzDavid/ICDAR-2019-SROIE

Steps:

https://nanonets.com/blog/information-extraction-graph-convolutional-networks/?&utm_source=nanonets.com/blog/&utm_medium=blog&utm_content=Adrian%20Sarno%20-%20AI%20&%20Machine%20Learning%20Blog


References:<br>
[Graph Neural Networks:
A Review of Methods and Applications](https://arxiv.org/pdf/1812.08434.pdf )

https://arxiv.org/pdf/1609.02907.pdf<br>
https://towardsdatascience.com/a-tale-of-two-convolutions-differing-design-paradigms-for-graph-neural-networks-8dadffa5b4b0<br>
https://arxiv.org/pdf/1812.08434.pdf<br>
https://www.math.ucla.edu/~tao/preprints/fourier.pdf<br>
<br>
https://arxiv.org/pdf/1606.09375.pdf


Sources:


[PICK: Processing Key Information Extraction from Documents using Improved Graph Learning-Convolutional Networks](https://arxiv.org/pdf/2004.07464.pdf)

Pseudocode for the project:

```python
# ------------------------------------------------------------------------
### Runs graph modeling process
# ------------------------------------------------------------------------
def run_graph_modeler(normalized_dir, target_dir):
    print("Running graph modeler")
        
    img_files, word_files, _ = \
    get_normalized_filepaths(normalized_dir)
    
    for img_file, word_file in zip(img_files, word_files):
        # reads normalized data for one image
        img, word_areas, _ = load_normalized_example(img_file, word_file)

        # computes graph adj matrix for one image
        lines = line_formation(word_areas)
        width, height = cv_size(img)
        graph = graph_modeling(lines, word_areas, width, height)
        adj_matrix = form_adjacency_matrix(graph)

        # saves node features and graph
        save_graph(target_dir, img_file, graph, adj_matrix)

# -------------------------------------------------------------------------
#### Numeric Features (neighbour distances from graph)
# -------------------------------------------------------------------------
def get_word_area_numeric_features(word_area, graph):
    """
    Extracts numeric features from the graph: returns 4 floats
    between -1 and 1 that represent the the relative distance
    to the nearest neighbour in each of the four main directions.
    Fills the "undefined" values with the null distance (0)
    """
    edges = graph[word_area.idx]
    get_numeric = lambda x: x if x != UNDEFINED else 0
    rd_left   = get_numeric(edges["left"][1])
    rd_right  = get_numeric(edges["top"][1])
    rd_top    = get_numeric(edges["right"][1])
    rd_bottom = get_numeric(edges["bottom"][1])
    return [rd_left, rd_right, rd_top, rd_bottom]


# -------------------------------------------------------------------------
#### Model Runner 
# -------------------------------------------------------------------------

for i in range(1, FLAGS.num_hidden_layers):
        in_dim = FLAGS.hidden1 * 2**(i-1)
        out_dim = in_dim * 2
        self.layers.append(GraphConvolution(input_dim=in_dim,
                                            output_dim=out_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging))


# -------------------------------------------------------------------------
#### GCN Layer forward transform
# -------------------------------------------------------------------------

# convolve
supports = list()
for i in range(len(self.support)):
pre_sup = dot(x, self.vars['weights_' + str(i)])
support = dot(self.support[i], pre_sup, sparse=True)
supports.append(support)
output = tf.add_n(supports)



# -------------------------------------------------------------------------
#### Global parameters controlling the execution
# -------------------------------------------------------------------------

from collections import namedtuple

# named tuples
WordArea = namedtuple('WordArea', 'left, top, right, bottom, content, idx')
TrainingParameters = namedtuple("TrainingParameters", "dataset model learning_rate epochs hidden1 num_hidden_layers dropout weight_decay early_stopping max_degree data_split")

# trainig parameters:
# --------------------------------------------------------------------------------
# dataset:        Selectes the dataset to run on
# model:          Defines the type of layer to be applied
# learning_rate:  Initial learning rate.
# epochs:         Number of epochs to train.
# hidden1:        Number of units in hidden layer 1.
# dropout:        Dropout rate (1 - keep probability).
# weight_decay:   Weight for L2 loss on embedding matrix.
# early_stopping: Tolerance for early stopping (# of epochs).
# max_degree:     Maximum Chebyshev polynomial degree.
FLAGS = TrainingParameters('receipts', 'gcnx_cheby', 0.001, 200, 16, 2, 0.6, 5e-4, 10, 3, [.4, .2, .4])

# output classes
UNDEFINED="undefined"
CLASSES = ["company", "date", "address", "total", UNDEFINED]

During data preprocessing, a function is called to calculate the Chebyshev approximation coefficients, these are computed from the adj matrix.



# -------------------------------------------------------------------------
#### During data preprocessing, a function is called to calculate the Chebyshev approximation coefficients, these are computed from the adj matrix.
# -------------------------------------------------------------------------

if FLAGS.model == 'gcnx_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCNX


# -------------------------------------------------------------------------
#### To print it in json format
# -------------------------------------------------------------------------




```

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment

    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py



--------
