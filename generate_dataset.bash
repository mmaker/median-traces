#!/bin/bash

# expecting UCIDv2 dataset.download from
# <http://homepages.lboro.ac.uk/~cogs/datasets/ucid/data/ucid.v2.tar.gz>
DATASET="$1"
COMPRESSION_LEVEL="$2"

PARALLEL="parallel"
ANTI="./contrib/run_fuck_median.sh /usr/local/MATLAB/R2014a/"
ANTI_THR=5

DOMAIN="-colorspace gray"
LUMINANCE_DOMAIN=""
MEDIAN3="-median 3"
MEDIAN5="-median 5"
GAUSSIAN="-gaussian-blur 0.5"
AVERAGE="-define convolve:scale=! -morphology Convolve square:3"
COMPRESS="-format jpg -quality $COMPRESSION_LEVEL%"
CROP="-gravity center -crop 128x128+0+0"
CROP_64="-gravity center -crop 64x64+0+0"
CROP_256="-gravity center -crop 256x256+0+0"
RESIZE="-resize 110% -resize 128x128"
RESIZE_64="-resize 110% -resize 64x64"
RESIZE_256="-resize 110% -resize 256x256"


UCID_DIR="dataset-j$COMPRESSION_LEVEL/ucid"
mkdir -p $UCID_DIR
tar xvf $DATASET -C $UCID_DIR
cd $UCID_DIR

make_dataset()
{
    echo "$1"

    dataset_dir="../$1"
    pattern="$2"
    args="${@:2:$#}"
    [ -d "$dataset_dir" ] || mkdir "$dataset_dir"
    find . -name "$PATTERN" | $PARALLEL convert {} $args "$dataset_dir/{.}.jpg"
}

make_antidataset()
{
    echo "$1-anti"

    src_dir="../$1"
    dst_dir="../$1-anti"
    mkdir "$dst_dir"
    $ANTI $PATTERN $src_dir $dst_dir $ANTI_THR
}

# 128x128 images, with jpeg compression.

PATTERN='*.tif'
make_dataset ori $CROP $DOMAIN           $COMPRESS
make_dataset mf3 $CROP $DOMAIN $MEDIAN3  $COMPRESS
make_dataset mf5 $CROP $DOMAIN $MEDIAN5  $COMPRESS
make_dataset gau $CROP $DOMAIN $GAUSSIAN $COMPRESS
make_dataset res $CROP $DOMAIN $RESIZE   $COMPRESS
make_dataset ave $CROP $DOMAIN $AVERAGE  $COMPRESS

PATTERN='*[0-7][0-9][02468].tif'
make_dataset mf35 $CROP $DOMAIN $MEDIAN3 $COMPRESS
PATTERN='*[0-7][0-9][13579].tif'
make_dataset mf35 $CROP $DOMAIN $MEDIAN5 $COMPRESS

PATTERN='ucid00[0-1]*.tif'
make_dataset all $CROP $DOMAIN           $COMPRESS
PATTERN='ucid00[2-3]*.tif'
make_dataset all $CROP $DOMAIN $AVERAGE  $COMPRESS
PATTERN='ucid00[4-5]*.tif'
make_dataset all $CROP $DOMAIN $GAUSSIAN $COMPRESS
PATTERN='ucid00[6-7].tif'
make_dataset all $CROP $DOMAIN $RESIZE   $COMPRESS

# 64x64 images, with jpeg compression

make_dataset ori-64 $CROP_64 $DOMAIN $COMPRESS

PATTERN='*[0-7][0-9][02468].tif'
make_dataset mf35-64 $CROP_64 $DOMAIN $MEDIAN3 $COMPRESS
PATTERN='*[0-7][0-9][13579].tif'
make_dataset mf35-64 $CROP_64 $DOMAIN $MEDIAN5 $COMPRESS

PATTERN='ucid00[0-1]*.tif'
make_dataset all-64 $CROP_64 $DOMAIN            $COMPRESS
PATTERN='ucid00[2-3]*.tif'
make_dataset all-64 $CROP_64 $DOMAIN $AVERAGE   $COMPRESS
PATTERN='ucid00[4-5]*.tif'
make_dataset all-64 $CROP_64 $DOMAIN $GAUSSIAN  $COMPRESS
PATTERN='ucid00[6-7].tif'
make_dataset all-64 $CROP_64 $DOMAIN $RESIZE_64 $COMPRESS
PATTERN='*.tif'

# 256x256 images, with jpeg compression

make_dataset ori-256  $CROP_256 $DOMAIN $COMPRESS

PATTERN='*[0-7][0-9][02468].tif'
make_dataset mf35-256 $CROP_256 $DOMAIN $MEDIAN3 $COMPRESS
PATTERN='*[0-7][0-9][13579].tif'
make_dataset mf35-256 $CROP_256 $DOMAIN $MEDIAN5 $COMPRESS

PATTERN='ucid00[0-1]*.tif'
make_dataset all-256 $CROP_256 $DOMAIN             $COMPRESS
PATTERN='ucid00[2-3]*.tif'
make_dataset all-256 $CROP_256 $DOMAIN $AVERAGE    $COMPRESS
PATTERN='ucid00[4-5]*.tif'
make_dataset all-256 $CROP_256 $DOMAIN $GAUSSIAN   $COMPRESS
PATTERN='ucid00[6-7]*.tif'
make_dataset all-256 $CROP_256 $DOMAIN $RESIZE_256 $COMPRESS


# anti dataset
PATTERN='*.jpg'
make_antidataset mf3
make_antidataset mf5
make_antidataset mf35
make_antidataset mf35-64
make_antidataset mf35-256
