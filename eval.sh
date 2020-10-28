#!/bin/sh

# $1 - dataset
# $2 - dataset name

for i in {1..10}
do
	python3 src/prepare_datasets.py --dataset $1 --name $2
	python3 main.py --name $2 --train_datasets $2 --test_dataset $2 --arch fcn
	python3 main.py --name $2 --train_datasets $2 --test_dataset $2 --arch fcn \
		--test --load checkpoint/$2/ckpt_best.pth.tar
done

