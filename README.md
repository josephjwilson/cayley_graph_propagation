# Cayley Graph Propagation

In this repository, we provide the code for Cayley Graph Propagation (**CGP**). We release our work under the MIT license.

*Note*: this is an active repository, therefore in the following days I will be uploading the full repository used to recreate the results as in our CGP paper.

## TL;DR
Cayley Graph Propagation (**CGP**) is a unqiue paper that addresses the [*over-squashing*](https://arxiv.org/abs/2111.14522) through a novel model that does not require *dedicated preprocessing*. We use the deserible botleneck-free graph structure, known as Cayley graphs as from [EGP](https://arxiv.org/abs/2210.02997). However, in CGP we leverage the complete Cayley graph structure that may not align perfectly with a corresponding input graph. Therefore, we have to handle the additional nodes as *virtual nodes*. In this repository we will illustrate how that is done!

## Getting started
To have an understanding of CGP we have created a starter guide in the form of an easy to follow [Google Colab](https://github.com/josephjwilson/cayley_graph_propagation/blob/main/Cayley_Graph_Propagation.ipynb). We hope this stand-alone file [Google Colab](https://github.com/josephjwilson/cayley_graph_propagation/blob/main/Cayley_Graph_Propagation.ipynb) provides an entry point for new GNN practitioner, or those simply wanted to implement CGP in their own models. The Google Colab shows how ease of modifying any repository to use CGP.

### Acknowledgements

In order to build this repository we would leveraged the open-source provided by following authors: [OGB](https://github.com/snap-stanford/ogb), [FoSR](https://github.com/kedar2/FoSR), and [LRGB](https://github.com/vijaydwivedi75/lrgb). Accordingly, we would also like to thank them for making our development process easier. 
