# TF-GNN: TensorFlow Graph Neural Networks

<!-- PLACEHOLDER FOR OVERVIEW GOOGLE EXTRAS -->

The TensorFlow GNN library makes it easy to build Graph Neural Networks, that
is, neural networks on graph data (nodes and edges with arbitrary features).
It provides TensorFlow code for building GNN models as well as tools for
preparing their input data and running the training.

Throughout, TF-GNN supports *heterogeneous* graphs, that is, graphs consisting
of multiple sets of nodes and multiple sets of edges, each with their own set of
features. These come up naturally when modeling different types of objects
(nodes) and their different types of relations (edges).


## User Documentation

Start with our introductory guides:

  * [Introduction to Graph Neural Networks](intro.md). This page introduces the
    concept of graph neural networks with a focus on their application at scale.

  * [The GraphTensor type](graph_tensor.md). This page introduces the
    `tfgnn.GraphTensor` class, which defines our representation of graph data
    in TensorFlow. We recommend that every user of our library understands its
    basic data model.

  * [Describing your graph](schema.md). This page explains how to declare the
    node sets and edge sets of your graph, including their respective features,
    with the `GraphSchema`
    [protocol message](https://developers.google.com/protocol-buffers).
    This defines the interface between data preparation (which creates such
    graphs) and the GNN model written in TensorFlow (which consumes these
    graphs as training data).

  * [Data preparation and sampling](data_prep.md). Training data for GNN
    models are graphs. This document describes their encoding as `tf.Example`s.
    Moreover, it introduces subgraph sampling for turning one huge graph into a
    stream of training inputs. TF-GNN offers two ways to run sampling:

      * The [In-Memory Sampler](inmemory_sampler.md) lets you run graph
        sampling on a single machine from main memory. Start here for an
        easy demo.
      * The [Beam Sampler](beam_sampler.md) lets you run distributed
        graph sampling, which scales way beyond in-memory sampling.

  * The [TF-GNN Runner](runner.md) lets you train GNN models on the
    prepared input data for a variety of tasks (e.g., node prediction).
    We recommend using the Runner to get started quickly with a first model
    for the data at hand, and then customize it as needed.

The following docs go deeper into particular topics.

  * The [Input pipeline](input_pipeline.md) guide explains how to set up
    a `tf.data.Dataset` for bulk input of the training and validation datasets
    produced by the [data preparation](data_prep.md) step. The TF-GNN Runner
    already takes care of this for its users.

  * [TF-GNN modeling](gnn_modeling.md) explains how to build a Graph Neural
    Network with TensorFlow and Keras, using the GraphTensor data from the
    previous steps. The TF-GNN library provides both a collection of standard
    models and a toolbox for writing your own. Users of the TF-GNN Runner
    are encouraged to consult this page to define custom models in the Runner.

  * The [Model saving](model_saving.md) guide covers technical details of
    saving TF-GNN models. (Most users of TF/Keras 2.13+ should be fine calling
    `tf.keras.Model.export()` without looking here.)

  * The [Keras version config](keras_version.md) guide explains how to install
    and use Keras v2 with TF2.16 and above, which is required for TF-GNN.

## Colab Tutorials

These Colab notebooks run complete examples of building and training a TF-GNN
model on a Google server from within your browser.

  * [Molecular Graph
    Classification](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/intro_mutag_example.ipynb)
    trains a model for the MUTAG dataset (from the
    [TUDatasets](https://chrsmrrs.github.io/datasets/) collection) that consists
    of 188 small, homogeneous graphs representing molecules. This is a good
    first read to get acquainted with GNNs.
  * [Solving OGBN-MAG
    end-to-end](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/ogbn_mag_e2e.ipynb)
    trains a model on heterogeneous sampled subgraphs from the
    [OGBN-MAG](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) dataset (from
    Stanford's Open Graph Benchmark) that contains 1 million research papers,
    their authors, and other relations. This colab introduces the node
    classification task from sampled subgraphs as well as the nuts and bolts of
    training in parallel on multiple accelerators (GPU, TPU).
  * An [in-depth OGBN-MAG
    tutorial](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/ogbn_mag_indepth.ipynb)
    that solves OGBN-MAG again while demonstrating how users can exercise
    greater control over the GNN model definition and the training code.
  * [Learning shortest paths with
    GraphNetworks](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/graph_network_shortest_path.ipynb)
    demonstrates an Encoder/Process/Decoder architecture for predicting the
    edges of a shortest path, using an Graph Network with edge states.
    Take a look if you are interested in advanced modeling.

## API Reference

TF-GNN comes with
[reference documentation for its API](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/api_docs/README.md),
extracted from the source code.

## Developer Documentation

How to contribute to the TF-GNN library.

  * [CONTRIBUTING.md](https://github.com/tensorflow/gnn/blob/main/CONTRIBUTING.md)
    describes the process for open-source contributions.
  * The [Developer](developer.md) guide describes how to clone our github repo
    and install the tools and libraries required to build and run TF-GNN code.

## Papers

The following research paper describes the design of this library:

  * O. Ferludin et al.: [TF-GNN: Graph Neural Networks in
  TensorFlow](https://arxiv.org/abs/2207.03522), 2023.

## Blog posts

  * [Graph neural networks in TensorFlow](https://blog.tensorflow.org/2024/02/graph-neural-networks-in-tensorflow.html)
    (February 06, 2024) for release 1.0.
  * [Introducing TensorFlow Graph Neural Networks](https://blog.tensorflow.org/2021/11/introducing-tensorflow-gnn.html)
    (November 18, 2021) for the initial open-sourcing ahead of release 0.1.

