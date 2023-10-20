# -*- coding: utf-8 -*-

import numpy as np
import pickle
import json


def load_sentic_word():
    path='/home/a303/document/sxw/DGEIAN/DGEIAN-gts+adj+rpd/senticNet/senticnet_word_0.4.txt'
    senticNet={}
    fp=open(path,'r')
    for line in fp:
        line=line.strip()
        if not line:
            continue
        word,sentic=line.split('\t')
        senticNet[word]=sentic
    fp.close()
    return senticNet


def affective_ck(text,senticNet):
    words=text.split()
    seq_len=len(words)
    matrix=np.zeros((seq_len,seq_len)).astype('float32')

    for i in range(seq_len):
        word=words[i].lower()

        if word in senticNet:
            sentic=float(senticNet[word])+1.0
        else:
            sentic=0
        for j in range(seq_len):
            matrix[i][j]=sentic
            matrix[j][i]=sentic
    for i in range(seq_len):
        if matrix[i][i]==0:
            matrix[i][i]=1

    return matrix


def process(filename):
    senticNet=load_sentic_word()
    idx2sentic={}

    fout=open(filename+'.sentic','wb')
    sentence_packs=json.load(open(filename))
    for sentence_pack in sentence_packs:
        sentic=affective_ck(sentence_pack['sentence'],senticNet)
        idx2sentic[sentence_pack['id']]=sentic

    pickle.dump(idx2sentic,fout)
    fout.close()


if __name__ == '__main__':
    process('/home/a303/document/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/ASTE_DATA_V2/lap14/train.json')
    process('/home/a303/document/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/ASTE_DATA_V2/lap14/test.json')
    process('/home/a303/document/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/ASTE_DATA_V2/lap14/dev.json')
    process('/home/a303/document/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/ASTE_DATA_V2/res14/train.json')
    process('/home/a303/document/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/ASTE_DATA_V2/res14/test.json')
    process('/home/a303/document/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/ASTE_DATA_V2/res14/dev.json')
    process('/home/a303/document/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/ASTE_DATA_V2/res15/train.json')
    process('/home/a303/document/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/ASTE_DATA_V2/res15/test.json')
    process('/home/a303/document/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/ASTE_DATA_V2/res15/dev.json')
    process('/home/a303/document/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/ASTE_DATA_V2/res16/train.json')
    process('/home/a303/document/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/ASTE_DATA_V2/res16/test.json')
    process('/home/a303/document/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/ASTE_DATA_V2/res16/dev.json')