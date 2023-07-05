# Text-Editing-AS-Imitation-Game

This repository is for the paper [Text Editing as Imitation Game](https://aclanthology.org/2022.findings-emnlp.114/). In *Findings of the Association for Computational Linguistics: EMNLP 2022*, pages 1583–1594, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

[[arXiv](https://arxiv.org/abs/2210.12276)] [[Poster](https://www.shininglab.ai/assets/posters/Text%20Editing%20as%20Imitation%20Game.pdf)] [[Slides](https://www.shininglab.ai/assets/slides/Text%20Editing%20as%20Imitation%20Game.pdf)] [[Video](https://www.youtube.com/watch?v=YwOcrWyRbos)]

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
@inproceedings{shi-etal-2022-text,
    title = "Text Editing as Imitation Game",
    author = "Shi, Ning  and
      Tang, Bin  and
      Yuan, Bo  and
      Huang, Longtao  and
      Pu, Yewen  and
      Fu, Jie  and
      Lin, Zhouhan",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.114",
    pages = "1583--1594",
    abstract = "Text editing, such as grammatical error correction, arises naturally from imperfect textual data. Recent works frame text editing as a multi-round sequence tagging task, where operations {--} such as insertion and substitution {--} are represented as a sequence of tags. While achieving good results, this encoding is limited in flexibility as all actions are bound to token-level tags. In this work, we reformulate text editing as an imitation game using behavioral cloning. Specifically, we convert conventional sequence-to-sequence data into state-to-action demonstrations, where the action space can be as flexible as needed. Instead of generating the actions one at a time, we introduce a dual decoders structure to parallel the decoding while retaining the dependencies between action tokens, coupled with trajectory augmentation to alleviate the distribution shift that imitation learning often suffers. In experiments on a suite of Arithmetic Equation benchmarks, our model consistently outperforms the autoregressive baselines in terms of performance, efficiency, and robustness. We hope our findings will shed light on future studies in reinforcement learning applying sequence-level action generation to natural language processing.",
}
```
