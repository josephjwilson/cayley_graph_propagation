# Graph Rewiring Module

This module provides implementations of various graph rewiring strategies that
can be used to enhance message passing in graph neural networks. The rewiring
algorithms modify the connectivity structure of the input graph while preserving
key properties.

## Architecture

The rewiring module is organized with the following components:

- `base.py`: Contains the `RewireStrategy` abstract base class that all rewiring strategies must implement
- `factory.py`: Provides a factory class for instantiating rewiring strategies by name
- Individual strategy implementations (e.g., `expander.py`, `fully_adjacent.py`, `fosr.py`, `digl.py`, `sdrf.py`, `borf.py`, `gtr.py`)

## Available Rewiring Strategies

Currently supported rewiring methods:

1. **Expander Graph Propagation (EGP)** - Uses Cayley graphs as expander structures, truncated to match the input graph size
2. **Cayley Graph Propagation (CGP)** - Uses complete Cayley graphs and marks extra nodes as virtual
3. **Fully Adjacent** - Creates a fully connected graph (complete graph)
4. **First-Order Spectral Rewiring (FoSR)** - Optimizes edge placement based on spectral properties of the graph to improve message passing
5. **Diffusion Improves Graph Learning (DIGL)** - Uses Personalized PageRank to rewire graphs based on diffusion dynamics
6. **Stochastic Discrete Ricci Flow (SDRF)** - Optimizes graph structure using discrete Ricci curvature to address oversquashing
7. **Balanced Optimal Ricci Flow (BORF)** - Uses Ollivier-Ricci curvature to identify negatively curved edges and add new shortcut edges
8. **Graph Traversal Rewiring (GTR)** - Adds edges that decrease the sum of effective resistances between nodes

## How to Add a New Rewiring Strategy

To add a new rewiring strategy:

1. Create a new Python file in the `rewiring` directory
2. Implement the `RewireStrategy` interface by subclassing `RewireStrategy`
3. Implement the required `rewire` method
4. Register your strategy with the `RewireFactory`

### Example

```python
# my_strategy.py
from rewiring.base import RewireStrategy
import torch
from typing import Optional

class MyCustomRewiring(RewireStrategy):
    def __init__(self, param=None):
        self.param = param
        
    def rewire(self, num_nodes: int, original_edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Implement your rewiring logic here
        # ...
        return new_edge_index

# Register with factory
from rewiring.factory import RewireFactory
RewireFactory.register_strategy('MyCustom', MyCustomRewiring)
```

## Usage

To use a rewiring strategy in a PyTorch Geometric transform or model:

```python
from rewiring.factory import RewireFactory

# Get a strategy by name
strategy = RewireFactory.get_strategy('EGP')

# Apply rewiring to get a new edge index
new_edge_index = strategy.rewire(num_nodes=100, original_edge_index=original_edges)
```

## Strategy-Specific Notes

### FoSR Rewiring

The First-Order Spectral Rewiring (FoSR) strategy requires the original edge index of the graph:

```python
strategy = RewireFactory.get_strategy('FoSR')
# Configure parameters (optional)
strategy.num_iterations = 100
strategy.initial_power_iters = 10
# Must provide original_edge_index
new_edge_index = strategy.rewire(num_nodes=100, original_edge_index=original_edges)
```

FoSR works by iteratively adding edges to improve the spectral gap of the normalized Laplacian matrix, which can enhance message passing in graph neural networks.

### DIGL Rewiring

DIGL can be configured to either select the top-k neighbors or use a threshold:

```python
strategy = RewireFactory.get_strategy('DIGL')
# Option 1: Top-k neighbors
strategy.alpha = 0.1  # Teleport probability
strategy.k = 128      # Number of neighbors to keep per node
strategy.eps = None   # Disable threshold-based clipping

# Option 2: Threshold-based clipping
strategy.alpha = 0.1    # Teleport probability
strategy.k = None       # Disable top-k selection
strategy.eps = 0.01     # Keep edges with weight > eps

# Apply rewiring (requires original_edge_index)
new_edge_index = strategy.rewire(num_nodes=100, original_edge_index=original_edges)
```

### SDRF Rewiring

SDRF can be configured to control curvature-based optimization:

```python
strategy = RewireFactory.get_strategy('SDRF')
# Configure parameters
strategy.loops = 10            # Number of rewiring iterations
strategy.remove_edges = True   # Whether to also remove high-curvature edges
strategy.removal_bound = 0.5   # Minimum curvature for edge removal
strategy.tau = 1               # Temperature parameter for softmax sampling
strategy.is_undirected = False # Whether the graph is undirected

# Apply rewiring (requires original_edge_index)
new_edge_index = strategy.rewire(num_nodes=100, original_edge_index=original_edges)
```

### BORF Rewiring

BORF uses Ollivier-Ricci curvature to optimize message flow:

```python
strategy = RewireFactory.get_strategy('BORF')
# Configure parameters
strategy.loops = 10             # Number of rewiring iterations
strategy.remove_edges = True    # Whether to also remove high-curvature edges
strategy.removal_bound = 0.5    # Minimum curvature for edge removal
strategy.batch_add = 4          # Number of edges to add in each iteration
strategy.batch_remove = 2       # Number of edges to remove in each iteration
strategy.algorithm = 'borf3'    # Which BORF implementation to use ('borf2' or 'borf3')

# Apply rewiring (requires original_edge_index)
new_edge_index = strategy.rewire(num_nodes=100, original_edge_index=original_edges)
```

### GTR Rewiring

GTR adds edges to minimize the total effective resistance:

```python
strategy = RewireFactory.get_strategy('GTR')
# Configure parameters
strategy.num_edges = 10         # Number of edges to add to the graph
strategy.try_gpu = True         # Whether to use GPU acceleration if available

# Apply rewiring (requires original_edge_index)
new_edge_index = strategy.rewire(num_nodes=100, original_edge_index=original_edges)
``` 