#!/bin/bash

DPATH="$1"
sets=(mf3 mf5 mf35 mf35-64 mf35-256)
thresholds=$(seq 1 0.5 10)
LEARN="python median_traces.py learn --path=$DPATH"
TEST="python median_traces.py test --path=$DPATH"
MEASURE="python median_traces.py measure"

for s in "${sets[@]}"
do
    for t in $thresholds
    do
	antidir="$DPATH/anti-testcases/$s-$t/"
	mfdir="$DPATH/$s"

	accuracy=$($TEST $s ori --cls $s $antidir)
	psnr=$($MEASURE psnr $mfdir $antidir | cut -d ' ' -f 1)
	ssim=$($MEASURE ssim $mfdir $antidir | cut -d ' ' -f 1)
	echo $s $t $accuracy $psnr $ssim

    done
done
