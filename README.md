# Cayley Graph Propagation

In this repository, we provide the code for Cayley Graph Propagation (**CGP**). We release our work under the MIT license.

*Note*: I am making updates to this repository on a separate branch (`cgp-updates`), so that it serves as a better foundation to be able to CGP and future methods against other graph-rewiring techniques. This should hopefully unify future experimentation. If you want to get started quickly with **CGP** I recommended following the provided [Google Colab](https://github.com/josephjwilson/cayley_graph_propagation/blob/main/Cayley_Graph_Propagation.ipynb).

## TL;DR
Cayley Graph Propagation (**CGP**) is a unique paper that addresses the [*over-squashing*](https://arxiv.org/abs/2111.14522) through a novel model that does not require *dedicated preprocessing*. We use the desirable bottleneck-free graph structure, known as Cayley graphs as from [EGP](https://arxiv.org/abs/2210.02997). However, in CGP we leverage the complete Cayley graph structure that may not align perfectly with a corresponding input graph. Therefore, we have to handle the additional nodes as *virtual nodes*. In this repository we will illustrate how that is done!

## Getting started
To have an understanding of CGP we have created a starter guide in the form of an easy to follow [Google Colab](https://github.com/josephjwilson/cayley_graph_propagation/blob/main/Cayley_Graph_Propagation.ipynb). We hope this stand-alone file [Google Colab](https://github.com/josephjwilson/cayley_graph_propagation/blob/main/Cayley_Graph_Propagation.ipynb) provides an entry point for new GNN practitioner, or those simply wanted to implement CGP in their own models. The Google Colab shows how easy it is to modify any repository to use CGP.

## Running CGP
To easily run CGP we make use of config files with the base file found in `utils/config.py`. This file format should provide guidance if you want to be easily able to adjust a hyperparameter. Here is an example:

```python
# Running CGP on MUTAG for TUDataset baseline
python main.py --cfg configs/TUDataset/graph_rewiring/base/GIN/mutag.yaml
```

In addition, you can pass command line arguments to make alterations to an existing config file. Here, we update the number of layers, use `gnn.num_layers`. Refer to the aforementioned `utils/config.py` file to have an understanding of the syntax.

```python
# Running CGP on MUTAG for TUDataset baseline, but manually set the number of layers to 5
python main.py --cfg configs/TUDataset/graph_rewiring/base/GIN/mutag.yaml gnn.num_layers 5
```

Notably, it is worth noting that the TUDataset has a lot of stochasticity, therefore it makes reproducing the results found in our paper less reliable.

### Acknowledgements

In order to build this repository we would leveraged the open-source provided by following authors: [OGB](https://github.com/snap-stanford/ogb), [FoSR](https://github.com/kedar2/FoSR), and [LRGB](https://github.com/vijaydwivedi75/lrgb). Accordingly, we would also like to thank them for making our development process easier. 
