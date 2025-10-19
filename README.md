# Attention Is All U Need

This repository contains Jupyter notebooks that reproduce and explore the "Attention Is All You Need" Transformer architecture and related experiments.

## Repository contents

- Jupyter Notebooks: All code and experiments are provided as notebooks (.ipynb).

## Getting started

1. Clone the repository:

   git clone https://github.com/aniketkarkhelikar/attention-is-all-u-need.git
   cd attention-is-all-u-need

2. (Recommended) Create and activate a virtual environment:

   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate    # Windows

3. Install dependencies:

   If a requirements.txt is provided, run:

   pip install -r requirements.txt

   Otherwise, install common packages used in the notebooks, e.g.:

   pip install numpy torch matplotlib pandas jupyter

4. Launch Jupyter Notebook / Lab:

   jupyter notebook

   or

   jupyter lab

## Notebooks

Each notebook explores a different part of the Transformer model (training, attention visualization, toy tasks, etc.). Open the notebooks in Jupyter to run and read commentary.

## Notes

- These notebooks are intended for educational and experimental purposes.
- GPU is recommended for training experiments using PyTorch.

## Contributing

Contributions, issues, and feature requests are welcome. Please open an issue or submit a pull request.

## License

Specify a license for the project (e.g., MIT) or add a LICENSE file.