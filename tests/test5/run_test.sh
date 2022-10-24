#! /bin/bash

python ../../md_cat.py --annotate 2 --maxIter 500 -i input_tree.nwk -t input_sampling_times.txt -p 1 -v -o output_tree.nwk -j output_clock.txt --randSeed 1123 -b
