from setuptools import find_packages, setup

setup(
    name="DoublePendulum",
    author="Underactuated Lab DFKI Robotics Innovation Center Bremen",
    version="1.0.0",
    url="https://github.com/dfki-ric-underactuated-lab",
    packages=find_packages(),
    install_requires=[
        # general
        "numpy",
        "matplotlib",
        "pandas",
        "scipy",
        "sympy",
        "scikit-learn",
        "cma",
        "lxml",
        "mini-cheetah-motor-driver-socketcan",
        "moteus",
        "inputs",
        "tabulate",
        "filterpy",
        "dill",
        "argparse",
        "opencv-python",
        "pyyaml",
        "torch==2.0.1",
        "torchvision>=0.14.1",  # ensure compatibility with torch 1.13.1
        "tensorboard",
        "h5py",
        "moviepy",
        "protobuf>=3.20.2",
        "torchode",
        # c++ python bindings
        "cython<1.0.0",
    ],
    extras_require={
        "all": [
            "sphinx",
            "sphinx-rtd-theme",
            "numpydoc",
            "pytest",
            "lark",
            "drake",
            "meshcat",
            "gymnasium",
            "stable_baselines3==2.3.2",
            "shimmy",
            "jax",
            "jaxlib",
            "flax",
            "tqdm",
            "cloudpickle==3.0.0",
        ],
        "doc": ["sphinx", "sphinx-rtd-theme", "numpydoc"],
        "test": ["pytest", "lark"],
        "OC": ["drake", "meshcat"],
        "RL": [
            "gymnasium",
            "stable_baselines3==2.3.2",
            "shimmy",
            "jax",
            "jaxlib",
            "flax",
            "tqdm",
            "cloudpickle==3.0.0",
            "evotorch",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Stable",
        "Environment :: Console",
        "Intended Audience :: Academic Usage",
        "Programming Language :: Python",
    ],
)
