"""
Setup script for the Gomoku Benchmark package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gomoku-benchmark",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive benchmark for evaluating LLMs on Gomoku game analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/gomoku-benchmark",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Board Games",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "transformers": ["torch>=1.9.0", "transformers>=4.20.0", "accelerate>=0.20.0"],
        "dev": ["pytest>=6.0.0", "pytest-cov>=2.12.0", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "gomoku-benchmark=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt"],
    },
)