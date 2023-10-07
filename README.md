# Neural Network implementatoin from scratch
This repository implements `nn_from_scratch` module, which is a pet-project, which implements basic Neural Networks functionality from scratch, using only `numpy` module.

This repository was created for completion of *Introduction to Computer Vision* course at Innopolis Unviersity.

## Structure
``` bash
├── lab1_rescaling # Labs of the course, also examples & testing of the module functionality 
│   ...
├── labX_task_description
├── nn_from_scratch
│   ├── LICENSE
│   ├── nn_from_scratch
│   │   ├── __init__.py
│   │   ├── interfaces.py # Abstract classes 
│   │   ├── nodes.py      # Implementation of different nodes
│   │   ├── neurons.py    # Implementation of different neurons 
│   │   ├── optimizers.py # Implementation of different optimizers
│   └── setup.py
└── README.md
```

## Module Installation
```bash
pip install --upgrade module
cd nn_from_scratch
python3 -m build
pip install -e .
```

## Functionality
The module currently supports:
1. Nodes
   * `SoftMax`
   * `NormalizedSoftMax` (which is equivalient to SoftMax(x / x_max))
   * `ReLU` 
2. Neurons
   * `Linear`
3. Optimizers

## Examples
1. `lab4_backprop` contains tests and usage examples of `SoftMax` and `NormalizedSoftMax` nodes.
2. `lab5_gradient` contains tests and usage examples of `ReLU` and `Linear` nodes.
