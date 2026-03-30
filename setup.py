"""Backward-compatible setup.py for cortex-memory."""
from setuptools import setup, find_packages

setup(
    name="cortex-memory",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy>=1.24.0"],
    extras_require={
        "transformers": ["sentence-transformers>=2.2.0"],
        "dev": ["pytest>=7.0", "pytest-cov>=4.0"],
    },
    python_requires=">=3.10",
)
