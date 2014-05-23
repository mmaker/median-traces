#!/bin/bash

DATASET="$1"

mkdir -p dataset/ucid

tar xvf $DATASET -C dataset/ucid

cd dataset/ucid
#### 128x128 images #####

echo 'original image dataset'
mkdir ../ori
find . -name '*.tif' -exec convert {} -gravity center -crop 128x128+0+0 -colorspace Gray ../ori/{} \;
cd ../ori

echo '3x3 median filtering'
mkdir ../mf3
find . -name '*.tif' -exec convert {} -median 3 ../mf3/{} \;

echo '5x5 median filtering'
mkdir ../mf5
find . -name '*.tif' -exec convert {} -median 5 ../mf5/{} \;

echo 'gaussian blur with standard deviation 0.5'
mkdir ../gau
find . -name '*.tif' -exec convert {} -gaussian-blur 0.5 ../gau/{} \;

echo 'rescaling with the scaling factor of 1.1'
mkdir ../res
find . -name '*.tif' -exec convert {} -resize 110% -resize 128x128 ../res/{} \;

echo '3x3 average filtering'
mkdir ../ave
find . -name '*.tif' -exec convert {} -define convolve:scale=! -morphology Convolve square:3 ../ave/{} \;

echo '3x3 and 5x5 median filtering'
mkdir ../mf35
find . -name '*[0-8][0-9][02468].tif' -exec convert {} -median 3 ../mf35/{} \;
find . -name '*[0-8][0-9][13579].tif' -exec convert {} -median 5 ../mf35/{} \;

echo 'mixed'
cd ..
mkdir all
cp ori/ucid00[0-1]*.tif all/
cp res/ucid00[2-3]*.tif all/
cp gau/ucid00[4-5]*.tif all/
cp ave/ucid00[6-7]*.tif all/
cd all


cd ../ucid
##### 64x64 images ####

echo 'original image dataset 64x64'
mkdir ../ori-64
find . -name '*.tif' -exec convert {} -gravity center -crop 64x64+0+0 -colorspace Gray ../ori-64/{} \;
cd ../ori-64/

echo '3x3 and 5x5 median filtering'
mkdir ../mf35-64
find . -name '*[0-7][0-9][02468].tif' -exec convert {} -median 3 ../mf35-64/{} \;
find . -name '*[0-7][0-9][13579].tif' -exec convert {} -median 5 ../mf35-64/{} \;

echo 'mixed'
mkdir ../all-64
find . -name 'ucid00[0-1]*.tif' -exec convert {} ../all-64/{} \;
find . -name 'ucid00[2-3]*.tif' -exec convert {} -define convolve:scale=! -morphology Convolve square:3 ../all-64/{} \;
find . -name 'ucid00[4-5]*.tif' -exec convert {} -resize 110% -resize 128x128 ../all-64/{} \;
find . -name 'ucid00[6-7].tif' -exec convert {} -gaussian-blur 0.5 ../all-64/{} \;


cd ../ucid
##### 256x256 images ####

echo 'original image dataset 256x256'
mkdir ../ori-256
find . -name '*.tif' -exec convert {} -gravity center -crop 256x256+0+0 -colorspace Gray ../ori-256/{} \;
cd ../ori-256

echo '3x3 and 5x5 median filtering'
mkdir ../mf35-256
find . -name '*[0-7][0-9][02468].tif' -exec convert {} -median 3 ../mf35-256/{} \;
find . -name '*[0-7][0-9][13579].tif' -exec convert {} -median 5 ../mf35-256/{} \;

echo 'mixed'
mkdir ../all-256
find . -name 'ucid00[0-1]*.tif' -exec convert {} ../all-256/{} \;
find . -name 'ucid00[2-3]*.tif' -exec convert {} -define convolve:scale=! -morphology Convolve square:3 ../all-256/{} \;
find . -name 'ucid00[4-5]*.tif' -exec convert {} -resize 110% -resize 128x128 ../all-256/{} \;
find . -name 'ucid00[6-7]*.tif' -exec convert {} -gaussian-blur 0.5 ../all-256/{} \;

cd ..
#### JPEG compression ####
find . -name '*.tif' -exec convert {} -quality 70% {}.jpeg \;
rename 's/.tif.jpeg/.jpeg/' */*
rm */*.tif
