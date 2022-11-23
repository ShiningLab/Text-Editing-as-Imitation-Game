# Text-Editing-AS-Imitation-Game

This repository is for the paper [Text Editing as Imitation Game](https://arxiv.org/abs/2210.12276) in *Findings of the Association for Computational Linguistics: EMNLP 2022*.

[[Poster](https://github.com/ShiningLab/Text-Editing-as-Imitation-Game/blob/main/assets/Text%20Editing%20as%20Imitation%20Game%20Poster.pdf)] [[Slides](https://github.com/ShiningLab/Text-Editing-as-Imitation-Game/blob/main/assets/Text%20Editing%20as%20Imitation%20Game%20Slides.pdf)] [[Video](https://www.youtube.com/watch?v=YwOcrWyRbos)]

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
If everything goes well, there should be a similar progressing shown below.
```
Initializing Game Environment ...
Constructing Editops with Metric levenshtein ...
*Configuration*
device: cpu
random seed: 0
game: aor
src vocab size: 16
tgt vocab size: 18
model: lstm
trainable parameters: 13,184,580
max trajectory length: 6
max state sequence length: 10
max action sequence length: 2
if load check point: False
original train size: 7000
train size: 7000
valid size: 1500
test size: 1500
if sampling: True
batch size: 256
learning rate: 0.001

Train Loss:2.8253 LR:0.001000: 100%|█████████████████████████████████████| 27/27 [00:55<00:00,  2.05s/it]
Train Epoch 0 Total Step 27 Loss:4.3506 Token Acc:0.2886 Seq Acc:0.1972
 src: - 2 + 11 - 6 + 8 == 11
 tar: <done> <done>
 pred: <done> <done>
Valid Loss:2.7447 LR:0.001000: 100%|██████████████████████████████████████| 6/6 [00:03<00:00,  1.70it/s]
Valid Epoch 0 Total Step 27 Loss:2.6269 Token Acc:0.5951 Seq Acc:0.4076
 src: 2 * 2 6 9 6
 tar: <pos_3> /
 pred: <pos_1> -
Test Loss:2.5733 LR:0.001000: 100%|███████████████████████████████████████| 6/6 [00:03<00:00,  1.72it/s]
Test Epoch 0 Total Step 27 Loss:2.6115 Token Acc:0.5934 Seq Acc:0.4121
 src: 6 + 10 - 2 9 5
 tar: <pos_5> -
 pred: <pos_5> +
 ...
```

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com

## BibTex
```
TODO.
```
