#!/bin/bash

VOCAB_FILE='vocab_21k.json'
VOCAB_SIZE=''

ARG_COMPILE=--${2:-'no-compile'}
ARG_BACKEND=--backend=${3:-'inductor'}

# Download data to container
python download_data.py

# Test CPU version
#python run.py train \
#    --train-src=./data/train.ja \
#    --train-tgt=./data/train.en \
#    --dev-src=./data/dev.ja \
#    --dev-tgt=./data/dev.en \
#    --vocab=vocab_21k.json \
#    --vocab-size-src=21000 \
#    --src-vocab-model="src_21k"
#    $ARG_COMPILE 

# GPU Version

CUDA_VISIBLE_DEVICES=0 python run.py train \
    --train-src=/tmp/data/train.ja \
    --train-tgt=/tmp/data/train.en \
    --dev-src=/tmp/data/dev.ja \
    --dev-tgt=/tmp/data/dev.en \
    --vocab=vocab_40k.json \
    --vocab-size-src=40000 \
    --src-vocab-model="src_40k" \
    $ARG_COMPILE $ARG_BACKEND \
    --gpu --lr=5e-4 --patience=1 \
    --valid-niter=200 --batch-size=32 --dropout=.3