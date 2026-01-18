# Attention Is All You Need

A comprehensive implementation and learning project for building GPT-2 from scratch and exploring the fundamentals of language modeling, automatic differentiation, and neural networks. 

## Project Overview

This repository contains educational implementations of core concepts in deep learning and natural language processing, with a focus on: 

- Building automatic differentiation engines (micrograd) from first principles
- Understanding backpropagation through manual gradient calculations
- Implementing GPT-2 architecture from scratch
- Exploring language modeling techniques

## Contents

### Notebooks

#### cleaned.ipynb
A detailed walkthrough of building a micrograd-like automatic differentiation engine from scratch. This notebook covers: 

**1. Foundational Concepts**
- Understanding derivatives through the limit definition
- Partial derivatives for multi-variable functions
- Numerical verification of gradients

**2. Core Value Object**
- Implementation of the `value` class for storing scalars and gradients
- Tracking computation graphs with parent nodes and operations
- Operator overloading for addition and multiplication

**3. Forward Pass**
- Building expression trees from basic operations
- Creating computation graphs for complex expressions
- Visualizing computational flows with Graphviz

**4. Backpropagation**
- Manual gradient calculation using the chain rule
- Step-by-step walkthrough of gradient flow
- Understanding local and global derivatives

**5. Gradient Verification**
- Numerical gradient checking against analytical gradients
- Validation of backpropagation implementation

#### micrograd.ipynb
An alternative implementation exploring similar concepts with additional experimentation and variations.

## Technical Details

### Dependencies

- **Python 3.12+**
- **NumPy**:  Numerical computations
- **Matplotlib**: Plotting and visualization
- **Graphviz**:  Computation graph visualization

### Installation

1. Clone the repository:
```bash
git clone https://github.com/aniketkarkhelikar/attention-is-all-u-need.git
cd attention-is-all-u-need
```

2. Set up a virtual environment (recommended):
```bash
python3 -m venv . venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install numpy matplotlib graphviz
```

4. Install Graphviz system dependency:
   - **Ubuntu/Debian**: `sudo apt-get install graphviz`
   - **macOS**: `brew install graphviz`
   - **Windows**: Download from [graphviz.org](https://graphviz.org/download/)

## Usage

### Running the Notebooks

Start Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```

Open either `cleaned.ipynb` or `micrograd.ipynb` and run cells sequentially to follow the implementation.



## Key Concepts Demonstrated

### Automatic Differentiation

The implementation demonstrates how to:
- Build a computational graph during forward pass
- Propagate gradients backward using the chain rule
- Store and update gradients for each node

### Chain Rule Application

For a computation graph where L depends on intermediate variables: 
```
L = f(g(h(x)))
```

The gradient is computed as: 
```
dL/dx = (dL/df) * (df/dg) * (dg/dh) * (dh/dx)
```

### Gradient Descent Foundation

Understanding these concepts is essential for:
- Training neural networks
- Implementing custom layers and operations
- Debugging gradient flow issues
- Optimizing model performance

## Project Structure

```
attention-is-all-u-need/
├── README.md
├── cleaned.ipynb          # Main tutorial notebook
├── micrograd.ipynb        # Alternative implementation
├── autogit                # Git automation script
└── . gitignore
```

## Learning Path

1. **Start with cleaned.ipynb**:  Follow the structured tutorial from basics to backpropagation
2. **Understand derivatives**: Review the mathematical foundations before diving into code
3. **Visualize graphs**: Use the provided visualization functions to understand data flow
4. **Verify gradients**: Always check your analytical gradients against numerical approximations
5. **Experiment**:  Modify the code to build more complex expressions and operations

## Current Status

**Work in Progress**

## Educational Goals

This project aims to provide:
- **Deep understanding** of how neural networks learn
- **Hands-on experience** with gradient computation
- **Practical implementation** of theoretical concepts
- **Foundation** for building more complex models

## Resources and References

- "Attention Is All You Need" paper by Vaswani et al. 
- Andrej Karpathy's micrograd repository
- Wikipedia:  Derivative and Chain Rule
- Deep Learning textbook by Goodfellow, Bengio, and Courville

## Future Enhancements

Planned additions:
- Complete GPT-2 implementation
- Training pipeline with real datasets
- More activation functions and operations
- Batched operations and tensor support
- Optimization algorithms (SGD, Adam, etc.)
- Model evaluation and inference examples

## Contributing

This is primarily an educational project. 

## Notes

- This implementation prioritizes clarity over performance
- Not intended for production use
- Designed for learning and understanding fundamentals
- Comments and markdown cells provide detailed explanations

## Author

[aniketkarkhelikar](https://github.com/aniketkarkhelikar)

## Acknowledgments

- Inspired by Andrej Karpathy's educational content (https://karpathy.ai/zero-to-hero.html)
- Based on the groundbreaking "Attention Is All You Need" paper
- Thanks to the deep learning community for open-source resources
