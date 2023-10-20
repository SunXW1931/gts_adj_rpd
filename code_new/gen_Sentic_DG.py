# -*- coding: utf-8 -*-
import json

import numpy as np
import spacy
import torch
import pickle

from spacy.tokens import Doc


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def load_sentic_word():
    """
    load senticNet
    """
    path = '/media/a303/C07EC7757EC762B0/document/sxw/sxw/DGEIAN/DGEIAN-gts+adj+rpd/senticNet/senticnet_word_0.6.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = sentic
    fp.close()
    return senticNet


def sentic_dependency_adj_matrix(text, senticNet):
    tokens=nlp(text)
    words=text.split()
    seq_len=len(words)
    matrix=np.zeros((seq_len,seq_len)).astype('float32')
    assert seq_len==len(list(tokens))

    for token in tokens:
        token_lower=str(token).lower()
        if token_lower in senticNet:
            sentic = float(senticNet[token_lower]) + 1
        else:
            sentic=0
        matrix[token.i][token.i] = 1 * sentic
        for child in token.children:
            matrix[token.i][child.i] = 1 * sentic
            matrix[child.i][token.i] = 1 * sentic

    return matrix


def process(filename):
    senticNet=load_sentic_word()
    idx2sentic={}

    fout=open(filename+'.sentic_dg','wb')
    sentence_packs=json.load(open(filename))
    for sentence_pack in sentence_packs:
        sentic_dg=sentic_dependency_adj_matrix(sentence_pack['sentence'],senticNet)
        idx2sentic[sentence_pack['id']]=sentic_dg

    pickle.dump(idx2sentic,fout)
    fout.close()


if __name__ == '__main__':
    process('/media/a303/C07EC7757EC762B0/document/sxw/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/lap14/train.json')
    process('/media/a303/C07EC7757EC762B0/document/sxw/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/lap14/test.json')
    process('/media/a303/C07EC7757EC762B0/document/sxw/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/lap14/dev.json')
    process('/media/a303/C07EC7757EC762B0/document/sxw/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/res14/train.json')
    process('/media/a303/C07EC7757EC762B0/document/sxw/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/res14/test.json')
    process('/media/a303/C07EC7757EC762B0/document/sxw/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/res14/dev.json')
    process('/media/a303/C07EC7757EC762B0/document/sxw/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/res15/train.json')
    process('/media/a303/C07EC7757EC762B0/document/sxw/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/res15/test.json')
    process('/media/a303/C07EC7757EC762B0/document/sxw/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/res15/dev.json')
    process('/media/a303/C07EC7757EC762B0/document/sxw/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/res16/train.json')
    process('/media/a303/C07EC7757EC762B0/document/sxw/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/res16/test.json')
    process('/media/a303/C07EC7757EC762B0/document/sxw/sxw/DGEIAN/DGEIAN-gts+adj+rpd/data/res16/dev.json')