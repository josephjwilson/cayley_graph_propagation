# Cayley Graph Propagation

In this repository, we provide the code for Cayley Graph Propagation (**CGP**). We release our work under the MIT license.

*Note*: For the moment, I have limited this repository to just the supporting `Cayley_Graph_Propagation.ipynb` notebook. This provides a quick start guide for easily implementing **CGP** within any other repository. I am making updates to this repository on a separate branch (`cgp-updates`), so that it serves as a better foundation to be able to use CGP and future methods against other graph-rewiring techniques. This should hopefully unify all future experimentation based on the invaluable repositories that **CGP** was built upon below. If you want to get started quickly with **CGP** I recommended the provided [Google Colab](https://github.com/josephjwilson/cayley_graph_propagation/blob/main/Cayley_Graph_Propagation.ipynb).

## TL;DR
Cayley Graph Propagation (**CGP**) is a unique paper that addresses the [*over-squashing*](https://arxiv.org/abs/2111.14522) through a novel model that does not require *dedicated preprocessing*. We use the desirable bottleneck-free graph structure, known as Cayley graphs as from [EGP](https://arxiv.org/abs/2210.02997). However, in CGP we leverage the complete Cayley graph structure that may not align perfectly with a corresponding input graph. Therefore, we have to handle the additional nodes as *virtual nodes*. In this repository we will illustrate how that is done!

## Getting started
To have an understanding of CGP we have created a starter guide in the form of an easy to follow [Google Colab](https://github.com/josephjwilson/cayley_graph_propagation/blob/main/Cayley_Graph_Propagation.ipynb). We hope this stand-alone file [Google Colab](https://github.com/josephjwilson/cayley_graph_propagation/blob/main/Cayley_Graph_Propagation.ipynb) provides an entry point for new GNN practitioner, or those simply wanted to implement CGP in their own models. The Google Colab shows how easy it is to modify any repository to use CGP.

### Acknowledgements

In order to build this repository we would leveraged the open-source provided by following authors: [OGB](https://github.com/snap-stanford/ogb), [FoSR](https://github.com/kedar2/FoSR), and [LRGB](https://github.com/vijaydwivedi75/lrgb). Accordingly, we would also like to thank them for making our development process easier. The listed repositories were used as the foundations to be able to easily conduct the experimentation for **CGP**, so many thanks for their open-source repositories.
