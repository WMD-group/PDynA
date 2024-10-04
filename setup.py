
from setuptools import find_packages, setup

with open("README.md") as file:
    long_description = file.read()

setup(
    name="pdyna",
    version="1.1.1",
    description="Perovskite dynamics analysis package",
    url="https://github.com/WMD-group/PDynA",
    author="Xia Liang",
    author_email="xia.liang16@imperial.ac.uk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Operating System :: OS Independent",
    ],
    keywords="perovskite dynamics analysis",
    test_suite="nose.collector", 
    packages=find_packages(),
    # Specify any non-python files to be distributed with the package
    package_data={'pdyna': ['basis/*.json']},  # include json files in the basis directory
    #package_data={
    #    "pdyna": ["pdyna/basis/*"],
    #},
    include_package_data=True,
    install_requires=[
        "scipy",
        "numpy",
        "matplotlib",
        "pymatgen",
        "ase",
    ],
    data_files=["LICENSE"],
)

