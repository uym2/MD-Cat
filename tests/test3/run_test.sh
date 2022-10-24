#! /bin/bash

b=50

python rtt.py input_tree.nwk input_sampling_times.txt > rtt_mu.txt
mu=`cat rtt_mu.txt`
maxbin=`echo 2*$mu | bc -l`
python get_bins.py $b $maxbin > input_clock.txt

python ../../run_emd.py --annotate 3 --maxIter 500 -i input_tree.nwk -t input_sampling_times.txt  -p 1 --clockFile input_clock.txt -v -o output_tree.nwk -j output_clock.txt --pseudo 1 --randSeed 1123
