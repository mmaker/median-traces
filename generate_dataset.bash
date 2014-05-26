#!/bin/bash

# expecting UCIDv2 dataset.download from
# <http://homepages.lboro.ac.uk/~cogs/datasets/ucid/data/ucid.v2.tar.gz>
DATASET="$1"

PARALLEL="parallel"

DOMAIN="-colorspace gray"
LUMINANCE_DOMAIN=""
MEDIAN3="-median 3"
MEDIAN5="-median 5"
GAUSSIAN="-gaussian-blur 0.5"
AVERAGE="-define convolve:scale=! -morphology Convolve square:3"
COMPRESS="-format jpg -quality 70%"
CROP="-gravity center -crop 128x128+0+0"
CROP_64="-gravity center -crop 64x64+0+0"
CROP_256="-gravity center -crop 256x256+0+0"
RESIZE="-resize 110% -resize 128x128"


mkdir -p dataset/ucid
tar xvf $DATASET -C dataset/ucid
cd dataset/ucid

PATTERN='*.tif'
make_dataset()
{
    dataset_dir="../$1"
    pattern="$2"
    args="${@:2:$#}"
    [ -d "$dataset_dir" ] || mkdir "$dataset_dir"
    find . -name "$PATTERN" | $PARALLEL convert {} $args "$dataset_dir/{.}.jpg"
}

echo 'original image dataset'
make_dataset ori $CROP $DOMAIN $COMPRESS
echo '3x3 median filtering'
make_dataset mf3 $CROP $DOMAIN $MEDIAN3 $COMPRESS
echo '5x5 median filtering'
make_dataset mf5 $CROP $DOMAIN $MEDIAN5 $COMPRESS
echo 'gaussian blur with standard deviation 0.5'
make_dataset gau $CROP $DOMAIN $GAUSSIAN $COMPRESS
echo 'rescaling with the scaling factor of 1.1'
make_dataset res $CROP $DOMAIN $RESIZE $COMPRESS
echo '3x3 average filtering'
make_dataset ave $CROP $DOMAIN $AVERAGE $COMPRESS
# mixed datasets
cd ..
echo '3x3 and 5x5 median filtering'
mkdir mf35
cp mf3/*[02468].jpg mf35/
cp mf5/*[13579].jpg mf35/
echo 'all!'
mkdir all
cp ori/ucid00[0-1]*.jpg all/
cp res/ucid00[2-3]*.jpg all/
cp gau/ucid00[4-5]*.jpg all/
cp ave/ucid00[6-7]*.jpg all/
cd ucid

##### 64x64 images ####
echo 'original image dataset 64x64'
make_dataset ori-64 $CROP_64 $DOMAIN $COMPRESS
echo '3x3 and 5x5 median filtering'
PATTERN='*[0-7][0-9][02468].tif'
make_dataset mf35-64 $CROP_64 $DOMAIN $MEDIAN3 $COMPRESS
PATTERN='*[0-7][0-9][13579].tif'
make_dataset mf35-64 $CROP_64 $DOMAIN $MEDIAN5 $COMPRESS
echo 'all-64'
PATTERN='ucid00[0-1]*.tif'
make_dataset all-64 $CROP_64 $DOMAIN $COMPRESS
PATTERN='ucid00[2-3]*.tif'
make_dataset all-64 $CROP_64 $DOMAIN $AVERAGE $COMPRESS
PATTERN='ucid00[4-5]*.tif'
make_dataset all-64 $CROP_64 $DOMAIN $GAUSSIAN $COMPRESS
PATTERN='ucid00[6-7].tif'
make_dataset all-64 $CROP_64 $DOMAIN $RESIZE $COMPRESS
PATTERN='*.tif'

# ##### 256x256 images ####
echo 'original image dataset 256x256'
make_dataset ori-256  $CROP_256 $DOMAIN $COMPRESS
echo '3x3 and 5x5 median filtering'
PATTERN='*[0-7][0-9][02468].tif'
make_dataset mf35-256 $CROP_256 $DOMAIN $MEDIAN3 $COMPRESS
PATTERN='*[0-7][0-9][13579].tif'
make_dataset mf35-256 $CROP_256 $DOMAIN $MEDIAN5 $COMPRESS
echo 'mixed'
PATTERN='ucid00[0-1]*.tif'
make_dataset all-256 $CROP_256 $DOMAIN $COMPRESS
PATTERN='ucid00[2-3]*.tif'
make_dataset all-256 $CROP_256 $DOMAIN $AVERAGE $COMPRESS
PATTERN='ucid00[4-5]*.tif'
make_dataset all-256 $CROP_256 $DOMAIN $GAUSSIAN $COMPRESS
PATTERN='ucid00[6-7]*.tif'
make_dataset all-256 $CROP_256 $DOMAIN $RESIZE $COMPRESS
PATTERN='*.tif'
