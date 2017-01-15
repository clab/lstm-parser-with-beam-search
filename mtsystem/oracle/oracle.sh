#!/bin/bash
reversed=oracle/dualign.py

while getopts ":r" opt; do
	case $opt in
		r)
			reversed=oracle/duralign.py
			python $reversed $1 $2 > out.txt
			python chudata.py < out123.txt > $3
			rm out123.txt
			;;
		\?)

			python $reversed $1 $2 > out123.txt
			python chudata.py < out123.txt > $3
			rm out123.txt
			;;
	esac
done