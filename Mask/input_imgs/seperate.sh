#!/bin/bash
cd raw/masked_faces/
for((i=0;i<100;i++));
do
  mv ${i}_* ../../train/masked
done
for((i=101;i<200;i++));
do
  mv ${i}_* ../../test_/masked
done
echo "finish masked"

cd ../unmasked_faces/
for((i=0;i<100;i++));
do
  mv ${i}_* ../../train/unmasked
done
for((i=101;i<200;i++));
do
  mv ${i}_* ../../test_/unmasked
done
echo "finish unmasked"
cd ../..
