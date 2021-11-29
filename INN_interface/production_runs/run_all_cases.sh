#!/bin/bash
for d in */; do
	if [[ $d =~ "_" ]]; then
		if [[ ! $d =~ "__" ]]; then
	    cd $d
			nohup python run_opt.py > out.txt 2>&1 &
			cd ..
		fi
	fi
done
