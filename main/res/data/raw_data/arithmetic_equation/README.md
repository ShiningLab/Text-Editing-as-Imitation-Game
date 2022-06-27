# Recurent-Text-Editing: Raw Data Generation

## Introduction
This folder contains code to generate raw datasets for the Arithmetic Operators Restoration (AOR), Arithmetic Equation Simplification (AES), and Arithmetic Equation Correction (AEC).

## Parameters
+ N - the number of unique integers
+ L - the number of integers in an equation
+ D - the number of unique equations

## Directory
+ **aor.py** - for AOR data generation
+ **aes.py** - for AES data generation
+ **aec.py** - for AEC data generation
+ **aor** - AOR raw datasets used in the original work
+ **aes** - AES raw datasets used in the original work
+ **aec** - AEC raw datasets used in the original work
```
data/
├── README.md
├── requirements.txt
├── aec
├── aec.py
├── aes
├── aes.py
├── aor
├── aor.py
└── utils.py
```

## Dependencies
+ python >= 3.7.7
+ numpy >= 1.18.4
+ tqdm == 4.46.0

## Run
Please note that the existing data will be overwritten by the new data. Run the following command to generate a dataset for AOR with:
+ N - 10
+ L - 5
+ D - 10000
```
$ python aor.py --N 10 --L 5 --D 10000
```

## Output
```
100%|██████████████████████████████████████████████████████████| 10000/10000 [00:10<00:00, 974.16it/s]
train size 7000 (7000, 2)
val size 1500 (1500, 2)
test size 1500 (1500, 2)
find output from aor/10N/5L/10000D
```

## Data Pre-processing
Please copy datasets to **code/main/res/data** for data pre-processing.
```
$ cd ..
$ cp -r data/aor main/res/data/aor/
$ cp -r data/aes main/res/data/aes/
$ cp -r data/aec main/res/data/aec/
```

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com