import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator,TabularDataset

import random
import math
import time

from Model3_function import tokenize_en
from Model3_Class import Encoder,Decoder,Seq2Seq
from Model3_Setting import Setting





##拿到数据
DOCUMENT = Field(sequential=True, tokenize=tokenize_en, lower=True,init_token='<sos>', eos_token='<eos>')
SUMMARY = Field(sequential=True, tokenize=tokenize_en, lower=True,init_token='<sos>', eos_token='<eos>')
fields=[("document",DOCUMENT),("summary",SUMMARY)]

train=TabularDataset(path=Setting.train_cvs,format="CSV",fields=fields,skip_header=True)
val=TabularDataset(path=Setting.val_cvs,format="CSV",fields=fields,skip_header=True)
test=TabularDataset(path=Setting.test_cvs,format="CSV",fields=fields,skip_header=True)


DOCUMENT.build_vocab(train, min_freq = 2)
SUMMARY.build_vocab(train, min_freq = 2)

#值作batch,便于迭代
device=Setting.device
BATCH_SIZE=Setting.BATCH_SIZE
train_iter = BucketIterator(train, batch_size=BATCH_SIZE, device=device, sort_key=lambda x: len(x.document), sort_within_batch=True)

val_iter =  BucketIterator(val, batch_size=BATCH_SIZE, device=device, sort_key=lambda x: len(x.document), sort_within_batch=True)

test_iter =  BucketIterator(test, batch_size=BATCH_SIZE, device=device, sort_key=lambda x: len(x.document), sort_within_batch=True)



##训练模型####################
##设置参数

INPUT_DIM = len(DOCUMENT.vocab)
OUTPUT_DIM = len(SUMMARY.vocab)
HID_DIM = Setting.ENC_EMB_DIM
ENC_LAYERS = Setting.ENC_LAYERS
DEC_LAYERS = Setting.DEC_LAYERS
ENC_HEADS = Setting.ENC_HEADS
DEC_HEADS = Setting.DEC_HEADS
ENC_PF_DIM = Setting.ENC_HID_DIM
DEC_PF_DIM = Setting.DEC_HID_DIM
ENC_DROPOUT = Setting.ENC_DROPOUT
DEC_DROPOUT = Setting.DEC_DROPOUT
LEARNING_RATE = Setting.lr
device=Setting.device

#实例化网络
enc = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device)

SRC_PAD_IDX = DOCUMENT.vocab.stoi[DOCUMENT.pad_token]
TRG_PAD_IDX = SUMMARY.vocab.stoi[SUMMARY.pad_token]


model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
print(model)

