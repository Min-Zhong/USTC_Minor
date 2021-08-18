#!/bin/bash
cd masked_people/
i=0
for imgs in *
do
    python ../detect.py $imgs ../masked_faces/${i}
    i=$[$i+1]
    echo $i
done
echo "finish masked"

cd ../
cd unmasked_people/
i=0
for imgs in *
do
    python ../detect.py $imgs ../unmasked_faces/${i}
    i=$[$i+1]
    echo $i
done
echo "finish unmasked"
cd ../
