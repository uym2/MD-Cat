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
MD-Cat accepts calibration points (hard constraints on divergence times) for internal nodes, sampling times at leaf nodes, and a mixture of the two. Below we give examples for the three most common use-cases. All examples are given in the folder [use_cases](use_cases) of this repository.

## Use case 1: Infer the unit ultrametric tree
If there is no calibration given, MD-Cat will infer the unit (depth 1) ultrametric tree.

``` bash
   python md_cat.py -i <INPUT_TREE> -o <OUTPUT_TREE>
```

We give an example in folder [use_cases/unit_time_tree](use_cases/unit_time_tree), inside which you can find the sampled input tree `input.nwk`. To try, use the following commands:

```bash
   cd use_cases/unit_time_tree
   python ../../md_cat.py -i input.nwk -o output.nwk -p 1 -v
```
The output tree is ```output.nwk```.
* It is an ultrametric tree and has depth (root-to-tip distance) 1.
* The relative divergence time and mutation rate of all internal nodes are annotated on the tree inside the square brackets with attribute `t`, as in, `[t=0.5261,mu=0.0764]`.

***Note: To reduce run time and show informative messages, we use the flags `-p 1` and `-v`. By default, `-p` is set to 100 and we recommend not reducing it to below 10 to allow a thorough optimization. Run `md_cat.py` with `-h` to learn more about these options.

## Use case 2: Infer the time tree from phylodynamics data
A typical use-case in virus phylogeny is to infer the time tree from a phylogram inferred from sequences and their sampling times (i.e. calibration points given at leaf nodes). MD-Cat reads the calibration points or sampling times from an input file via the `-t` option.

```bash
   python ../../md_cat.py -i <INPUT_TREE> -o <OUTPUT_TREE> -t <SAMPLING_TIMES>
```

### 2.1. Complete sampling times
An example is given in the folder `use_cases/virus_all_samplingTime`. Starting from the base directory,

```bash
   cd use_cases/virus_all_samplingTime
```

Inside this folder you will find an input tree (`input.nwk`) and the input sampling times (`input.txt`).
In this example, we give MD-Cat all the sampling times for all leaves (i.e. complete sampling times).

### Sampling time / calibration time file

* The sampling time file (`input.txt`) is a tab-delimited file, with one pair of species-time per line
* It must have two columns: the species names and the corresponding sampling times.

 For example, lines

```
000009  9.36668
000010  9.36668
000011  11.3667
000012  11.3667
```
show that leaves `000009` and `000010` are sampled at time 9.36668 while nodes `000011` and `000012` are sampled at time 11.3667.

**Note:** These times are assumed to be forward; i.e, smaller values mean closer to the root of the tree. The top of the branch above the root is assumed to be 0.

Now, run:

```bash
   cd use_cases/virus_all_samplingTime
   python ../../md_cat.py -i input.nwk -o output.nwk -t input.txt -p 1 -v
```
The output tree ```output.nwk```

* has branch lengths in time units and
* divergence times annotated on every internal nodes using the `[t=9.55]` notation.

### 2.2. Partial (missing) sampling times
MD-Cat allows missing sampling times for the leaves, as long as there exists at least one pair of leaves with different sampling times. The usage of MD-Cat is the same as in the case of complete sampling times. An example is given in the folder `use_cases/virus_some_samplingTime`. Here we give the sampling times for 52 species out of 110 in total.

```bash
   cd use_cases/virus_some_samplingTime/
   python ../../md_cat.py -i input.nwk -o output.nwk -t input.txt -p 1 -v
```

### 2.3. Sampling times at internal nodes
MD-Cat allows the sampling times to be given in both internal nodes and at leaves. An example is given in the folder `use_cases/virus_internal_smplTime`. In the example tree, each of the nodes (including leaf nodes and internal nodes) has a unique label. If the internal nodes have unique labeling, MD-Cat allows the internal calibrations to be specified by their labels such as the leaves.

```bash
   cd use_cases/virus_internal_smplTime
   python ../../md_cat.py -i input.nwk -o output.nwk -t input.txt -p 1 -v
```

### 2.4. Sampling times given in date format
MD-Cat also allows sampling times to be given in date format (yyyy-mm-dd) when used with `-d`. For example, lines

```
000072 2013-01-01
000073 2013-01-01
000090 2015-10-11
000091 2015-10-11
```
show that leaves `000072` and `000073` were sampled on Jan 1st 2013 while `000090` and `000091` were sampled on Oct 11th 2015. The input date must follows the correct format (yyyy-mm-dd or yyyy-mm). If the day is missing, it will be implicitly input to MD-Cat as the first day of the month. 

Now, run:

```bash
   cd use_cases/virus_date_samplingTime
   python ../../md_cat.py -i input.nwk -o output.nwk -t input.txt -d -p 1 -v
```
With `-d`, the branch lengths and mutation rates of the output tree are in the unit of years, and divergence times are in date format.

## Use case 3: Infer the time tree with calibration points given in backward time
For calibration points obtained from fossils, the calibration points are usually specified in backward time such as "million years ago" ("mya"). For these cases, MD-Cat allows specification of backward time via the `-b` flag.

```bash
   python ../../md_cat.py -i <INPUT_TREE> -o <OUTPUT_TREE> -t <CALIBRATIONS> -b
```
Calibration points can be given in the same way as sampling times.

* If the tree nodes are uniquely labeled, we can use the node labels to specify the internal nodes associated with the calibration points.
* Alternatively, MD-Cat allows the identification of a node as the Least Common Ancestor (LCA) of a set of species and allows optional label assignment to these calibration points. You may know LCA by their other name: MRCA.

We give an example of the LCA specification in `use_cases/fossil_backward_time`. From the base directory, go to this example.

```bash
   cd use_cases/fossil_backward_time
```

Because the input tree ```input.nwk``` does not have labels for internal nodes, we need to use LCA to specify calibration points. Here we use 4 calibration points in ```input.txt```:

```
Fagales 95.1
Myrtales=Terminalia+Eucalyptus 87.3
Archaefructus=Nuphar+Trithuria 126.5
Aquifoliaceae 66.5
```

An internal node can be identified by either its label or the LCA of 2 or more species separated by `+`. Moreover, a name for this internal node can be optionally specified using `=`. In our example, the 4 calibration points are: the internal node labeled as `Fagales`, the LCA of `(Terminalia and Eucalyptus)` that we wish to name `Myrtales`, the LCA of `(Nuphar and Trithuria)` that we wish to name `Archaefructus`, and the internal node labeled as `Aquifoliaceae`. Note that label assignments in ```input.txt``` will override the original input tree's labels.

```bash
   python ../../md_cat.py -i input.nwk -t input.txt -o output.nwk -b -p 1 -v
```

With the `-b` flag, MD-Cat understands the time as backward and enforces each parent node's divergence time to be larger (i.e. "older") than those of its children.

The output tree ```output.nwk``` is ultrametric, has branch lengths in time units, and has divergence times annotated onto the internal nodes in backward time.

* By default, the leaf nodes are set to present time (t = 0). You can adjust the leaf time using the `-f` option.
* The output tree has internal node labels the same as the input tree, except for the two calibration points "Myrtales" and "Archaefructus" assigned by user via `input.txt`.


# Other useful options

The following options are useful to explore:

* `-v` can be used to turn on the verbose mode.
* `-k 50` (or some other number) can be used to specify the number of rate categories. Default is 50.
* `-p 100` (or some other number) can be used to specify the number of times the optimization problem is solved, each starting from a different initial point. Default is 100. 
* `-l` can be used to set the length of the sequences from which the tree is inferred. Impacts the pseudocount used internally by MD-Cat for super short branches.
* `-r` and `-f` can be used to set the time at the root and the leaves.
* `--maxIter` to adjust the maximum number of iterations of the internal optimizer.
* `--randSeed` can be used to set the seed number, to enable reproducible results.
* `--annotate` can be used to set the level of annotation on the output tree. 
