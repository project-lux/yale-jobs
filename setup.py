from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="yale-jobs",
    version="0.0.1",
    author="William J.B. Mattingly",
    description="Job management for Yale's HPC cluster - like HuggingFace Jobs for Yale",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        # "pyyaml>=6.0",
        "paramiko>=4.0.0",
        
        # # Data processing
        # "datasets>=2.14.0",
        # "huggingface-hub>=0.17.0",
        # "pillow>=10.0.0",
        
        # # Data sources
        # "pypdfium2>=4.0.0",  # PDF processing
        # "requests>=2.31.0",   # Web/IIIF data
        
        # # ML/OCR (optional, but recommended)
        # "torch>=2.0.0",
        # "vllm>=0.5.0",
        # "tqdm>=4.65.0",
        # "toolz>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "ocr": [
            "torch>=2.0.0",
            "vllm>=0.5.0",
            "transformers>=4.30.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "yale=yale.cli:main",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
