import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyml",
    version="0.0.1",
    author="Denis Kotov (redradist, RedRadist, redra, RedRa)",
    author_email="redradist@gmail.com",
    description="PyML is a package for developing Machine Learning algorithms and Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)