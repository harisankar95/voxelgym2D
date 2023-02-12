from setuptools import find_packages, setup

setup(
    name="voxelgym2D",
    version="0.2",
    description="Gym environment for 2D grid path planning",
    author="Harisankar Babu",
    author_email="harisankar995@gmail.com",
    keywords="reinforcement-learning machine-learning gym openai python data-science",
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
    packages=find_packages(),
    package_data={
        "voxelgym2D": ["envs/maps/*.npy"],
    },
    install_requires=[
        # sb3 support for gym 0.21
        "gym==0.21",
        "numpy",
        "scikit-image",
        "opencv-python",
        "pathfinding==1.0.1",
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
            "tox",
            "sphinx",
            "sphinx_rtd_theme",
        ],
        "sb3": [
            "stable-baselines3[extra]==1.6.2",
            "sb3-contrib==1.6.2",
        ],
    },
    python_requires=">=3.7",
)
