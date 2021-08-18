#!/bin/bash
echo "Extracting faces from pictures downloaded"
echo "Begin extracting cigarette faces..."
rm -rf ./fig/faces/cigarette/train
mkdir -p ./fig/faces/cigarette/train
cd ./fig/pics/cigarette/
for pic in *
do
  python ../../../python/face_detect.py $pic ../../faces/cigarette/train/
done
cd ../../../
echo "Finished!"

rm -rf ./fig/faces/cigarette/test
mkdir ./fig/faces/cigarette/test
#cd ./fig/faces/cigarette/train/
#for((i=900;i<1200;i++)); 
#do  
#  mv ${i}_* ../test
#done
#cd ../../../../

echo "Begin extracing normal faces..."
rm -rf ./fig/faces/normal/train
mkdir -p ./fig/faces/normal/train
cd ./fig/pics/normal/
for pic in *
do
  python ../../../python/face_detect.py $pic ../../faces/normal/train/
done
cd ../../../
echo "Finished!"

rm -rf ./fig/faces/normal/test
mkdir ./fig/faces/normal/test
#cd ./fig/faces/normal/train/
#for((i=450;i<600;i++)); 
#do
#  mv ${i}_* ../test
#done
#cd ../../../../
