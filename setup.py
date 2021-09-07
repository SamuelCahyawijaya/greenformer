import os
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyAutoFact",
    version="0.1.14",
    author="Samuel Cahyawijaya",
    author_email="samuel.cahyawijaya@gmail.com",
    description="Auto Factorization package for PyTorch modules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SamuelCahyawijaya/py_auto_fact",
    project_urls={
        "Bug Tracker": "https://github.com/SamuelCahyawijaya/py_auto_fact/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "matrix-fact==1.1.2",
        "transformers==4.8.2",
        "torch>=1.5.0",
        "scipy",
        "cvxopt==1.2.6",
        "matplotlib==3.4.2",
        "seaborn==0.11.1",
        "sklearn"
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.3",
)
