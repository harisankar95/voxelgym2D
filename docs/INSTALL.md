# Installation

## PyPI

Install the package from PyPI using pip:

```bash
pip install voxelgym2D
```

## GitHub

```bash
pip install git+https://github.com/harisankar95/voxelgym2D.git
```

## For development purpose use editable mode

To install the package in development mode, run the following command in the root directory of the repository:

```bash
git clone https://github.com/harisankar95/voxelgym2D.git ~/path/to/repo
cd ~/path/to/repo
pip install -e .[dev]

# to aditionally install stable_baselines 3 and pytorch (optional)
pip install -e .[dev,sb3]
```
