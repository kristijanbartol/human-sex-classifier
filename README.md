# Gender classifier

## Introduction

An implementation for the "Can Human Sex Be Learned Using Only 2D Keypoint Estimations?" paper.

## Installation

Use `pip3 install requirements.txt` or `docker build -t sex-recognition .`.

## Prepare, train and evaluate

To run the training, first download ([PETA](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html) and/or [3DPeople](https://cv.iri.upc-csic.es/)) and prepare the datasets:

```
python3 src/prepare_datasets.py --name peta --dataset peta
```

Then you can run the training:

```
python3 main.py --name peta --train_datasets peta --test_dataset peta --arch fcn
```

To evaluate the model, run the experiments multiple times (as input data is small and the architecture is simple, it should take only few minutes per experiment):

```
./eval peta peta
```

To get the boxplots and the correlations from the paper, use the scripts from `report/` directory:

```
python3 correlation.py
python3 report.py peta
```

You can also combine multiple training datasets, for example:

```
python3 main.py --name peta --train_datasets 3dpeople,peta --test_dataset peta --arch fcn
```

See more data preparation and training options by:

```
python3 src/prepare_datasets.py -h
python3 main.py -h
```

## License

MIT
