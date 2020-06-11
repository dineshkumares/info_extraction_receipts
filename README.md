info_extraction_receipts
==============================

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




```
Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
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
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
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
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
