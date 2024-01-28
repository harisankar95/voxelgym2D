#!/usr/bin/env python
import os

from setuptools import find_packages, setup

# read the version from version.txt
with open(os.path.join("voxelgym2D", "version.txt"), encoding="utf-8") as file_handler:
    __version__ = file_handler.read().strip()

setup(
    name="voxelgym2D",
    version=__version__,
    description="Gym environment for 2D grid path planning",
    author="Harisankar Babu",
    author_email="harisankar995@gmail.com",
    keywords=["reinforcement-learning", "machine-learning", "gym", "openai", "python", "gymnasium"],
    license="MIT",
    url="https://github.com/harisankar95/voxelgym2D.git",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=[package for package in find_packages() if package.startswith("voxelgym2D")],
    package_data={
        "voxelgym2D": ["envs/maps/*.npy", "version.txt"],
    },
    install_requires=[
        "gymnasium<=0.29.1",
        "numpy",
        "scikit-image<=0.22.0",
        "opencv-python",
        "pathfinding>=1.0.9",
        # rendering
        "matplotlib",
    ],
    extras_require={
        "dev": [
            "pytest",
            "coverage",
            "pylint",
            "mypy",
            "isort",
            "black",
            "tox>=4.5.0",
            "sphinx",
            "sphinx_rtd_theme",
            "recommonmark",
            "sphinx-autodoc-typehints",
            "sphinx-copybutton",
            "sphinx-prompt",
            "sphinx-notfound-page",
            "sphinx-autodoc-annotation",
        ],
        "sb3": ["stable-baselines3[extra]>=2.0.0", "sb3-contrib>=2.0.0", "rl_zoo3>=2.0.0"],
    },
    python_requires=">=3.8",
    platforms=["any"],
)
