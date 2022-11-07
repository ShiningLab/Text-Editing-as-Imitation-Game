# Text-Editing-AS-Imitation-Game

This repository is for the paper [Text Editing as Imitation Game](https://arxiv.org/abs/2210.12276) in *Findings of the Association for Computational Linguistics: EMNLP 2022*.

## Directory
+ **main/env** - Game Environments
+ **main/res** - Resources including model check points, datasets, and experiment records
+ **main/src** - Source code including model structures, training pipelines, and utility functions
```
Text-Editing-as-Imitation-Game
├── README.md
├── assets
├── main
│   ├── config.py
│   ├── envs
│   ├── main.py
│   ├── res
│   │   ├── ckpts
│   │   ├── data
│   │   └── log
│   └── src
│       ├── models
│       ├── trainers
│       └── utils
└── requirements.txt
```

## Dependencies
+ python >= 3.10.6
+ tqdm >= 4.64.1
+ numpy >= 1.23.4
+ torch >= 1.13.0

## Setup
Please ensure required packages are already installed. A virtual environment is recommended.
```
$ cd Text-Editing-AS-Imitation-Game
$ cd main
$ pip install pip --upgrade
$ pip install -r requirements.txt
```

## Models
+ AR - base_seq2seq_lstm
+ AR* - seq2seq_lstm
+ NAR - base_lstm
+ D2 (NAR*) - lstm
+ Decoder0 - lstm0
+ Linear - lstm1
+ Shared D2 - lstm2

## Run
Before training, please take a look at the **config.py** to ensure training configurations.
```
$ vim config.py
$ python main.py
```

## Output
If everything goes well, there should be a similar progressing shown as below.
```
Initializing Game Environment ...
Constructing Editops with Metric levenshtein ...

*Configuration*
device: cuda
random seed: 0
...

Train Epoch 0 Total Step 27 Loss:2.2370 Token Acc:0.2269 Seq Acc:0.1189
 src: 4 + 7 - 6 - 2 == 3
 tar: <done> <done>
 pred: <done> +
...
```

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com

## BibTex
```
TODO.
```
