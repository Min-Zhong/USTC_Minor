#!/bin/bash
echo "Downloading pictures..."

echo "Downloading pictures of cigarette..."
rm -rf ./fig/pics/cigarette
mkdir ./fig/pics/cigarette
python ./python/pic_download.py 人抽烟 4 ./fig/pics/cigarette/
python ./python/pic_download.py 抽烟 4 ./fig/pics/cigarette/
python ./python/pic_download.py 吸烟 4 ./fig/pics/cigarette/

echo "Finished!"
echo "Downloading pictures of normal faces..."
rm -rf ./fig/pics/normal
mkdir ./fig/pics/normal
python ./python/pic_download.py 人脸 4 ./fig/pics/normal/
python ./python/pic_download.py 脸 4 ./fig/pics/normal/
python ./python/pic_download.py 人 4 ./fig/pics/normal/
