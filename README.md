# About

MD-Cat is a method for dating phylogenetic trees. Given a phylogeny and either sampling times for leaves or calibration points for internal nodes, MD-Cat outputs a "dated" tree that conforms to the sampling times or calibration points. MD-Cat relaxes the molecular clock assumption by approximating the rate distribution by a categorical distribution. The only required parameter for this model is the number of rate categories, which is default to 50.

#### Publication
Mai, Uyen, Eduardo Charvel, and Siavash Mirarab. “Expectation-Maximization enables Phylogenetic Dating under a Categorical Rate Model.” bioRxiv (2022).

#### Contact
Please submit questions and bug reports as [issues](https://github.com/uym2/MD-Cat/issues).

# Installation
MD-Cat has been tested on MacOS and Linux.

### Prerequisites

You need to have:
* Python >= 3.7
* The proper Mosek license for your machine. A free academic license can be obtained by visiting [the Mosek website](https://www.mosek.com/products/academic-licenses/)

### Install from source code

1. Download the source code.  
	* Either clone the repository to your machine 

	```bash
	   git clone https://github.com/uym2/MD-Cat.git
	```
	* or simply download [this zip file](https://github.com/uym2/MD-Cat/archive/master.zip) to your machine and unzip it in your preferred destination. 
2. To install, go to the MD-Cat folder. 
	* If you have ```pip```, use
	```bash
	   python3 -m pip install .
	```
	* Otherwise, type
	``` bash
	   python3 setup.py install
	```
After installation, run:

```bash
python3 md_cat.py -h
```
to see the commandline help of MD-Cat.

# Usage
MD-Cat accepts calibration points (hard constraints on divergence times) for internal nodes, sampling times at leaf nodes, and a mixture of the two. Below we give examples for the three most common use-cases. 

* All examples are given in the folder [tests](tests) of this repository.
