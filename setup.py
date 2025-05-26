#!/usr/bin/env python3
"""
Setup script for OpenBehavior.

Advanced LLM Behavior Analysis and Evaluation Platform
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="openbehavior",
    version="1.0.0",
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    description="Advanced LLM Behavior Analysis and Evaluation Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llamasearchai/OpenBehavior",
    project_urls={
        "Bug Tracker": "https://github.com/llamasearchai/OpenBehavior/issues",
        "Documentation": "https://github.com/llamasearchai/OpenBehavior#readme",
        "Source Code": "https://github.com/llamasearchai/OpenBehavior",
    },
    packages=find_packages(where="python"),
    package_dir={"": "python"},
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
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "server": [
            "uvicorn>=0.20.0",
            "gunicorn>=20.1.0",
        ],
        "dashboard": [
            "streamlit>=1.28.0",
            "plotly>=5.15.0",
            "pandas>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "openbehavior=openbehavior.cli.main:cli",
            "ob=openbehavior.cli.main:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "openbehavior": [
            "templates/*.yaml",
            "templates/*.json",
            "prompts/*.txt",
            "prompts/*.yaml",
        ],
    },
    keywords=[
        "ai",
        "llm",
        "behavior-analysis",
        "safety",
        "ethics",
        "alignment",
        "evaluation",
        "machine-learning",
        "natural-language-processing",
    ],
    zip_safe=False,
) 