import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyAutoFact",
    version="0.0.1",
    author="Samuel Cahyawijaya",
    author_email="samuel.cahyawijaya@gmail.com",
    description="Auto Factorization package for PyTorch modules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SamuelCahyawijaya/auto_fact",
    project_urls={
        "Bug Tracker": "https://github.com/SamuelCahyawijaya/auto_fact/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.3",
)
