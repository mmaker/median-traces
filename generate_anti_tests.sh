#!/bin/sh
# expecting a directory structure previously generated via generate_dataset.bash.

ROOT="$1"
ANTI="/home/maker/dev/uni/data-hiding/project/contrib/run_fuck_median.sh /usr/local/MATLAB/R2014a/"
PATTERN='*.jpg'

cd $ROOT
mkdir -p anti-testcases

make_antidataset()
{
    echo "$1-anti / threshold: $2"

    src_dir="$1"
    dst_dir="anti-testcases/$1-$2"
    mkdir -p "$dst_dir"
    $ANTI $PATTERN $src_dir $dst_dir $2
}


for thr in {5..10}
do
    make_antidataset mf5      $thr
    make_antidataset mf3      $thr
    make_antidataset mf35-64  $thr
    make_antidataset mf35-256 $thr
done
