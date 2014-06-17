#!/bin/bash

DPATH="$1"
sets=('mf3' 'mf5' 'mf35' 'mf35-64' 'mf35-256')
thresholds=$(seq 1 0.5 10)
TEST="python median_traces.py test --path=$DPATH"
MEASURE="python median_traces.py ssim"

for s in $sets
do
    for t in $thresholds
    do
	antidir="$DPATH/anti-testcases/$s-$t/"
	mfdir="$DPATH/$s"
	printf "ssim $s $t " && $MEASURE ssim $mfdir $antidir
    printf "psnr $s $t " && $MEASURE psnr $mfdir $antidir
    done
done
