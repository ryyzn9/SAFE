from setuptools import setup, find_packages

setup(
    name="safe",
    version="0.1.0",
    description="Ray Distributed Alignment Training with SAFE and Asymmetric KL algorithms",
    author="SAFE Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "datasets>=2.14.0",
        "ray[default]>=2.9.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "safe-train=scripts.train:main",
            "safe-distributed=scripts.train_distributed:main",
        ]
    },
)
