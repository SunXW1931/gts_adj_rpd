import json

import numpy as np
import spacy
import torch
import torch.nn.functional as F
import pickle
import math

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


# 进行标准化
def z_score(matrix,n):
    # 计算均值
    average=np.mean(matrix,axis=-1)
    variance=np.var(matrix)
    matrix_z=abs(matrix-average)
    matrix_z=matrix_z/variance
    return matrix_z


def relative_position_distance(text):
    tokens=nlp(text)
    words=text.split()
    seq_len=len(words)
    matrix=np.zeros((seq_len,seq_len)).astype('float32')
    assert len(list(tokens))==seq_len

    for i in range(seq_len):
        for j in range(seq_len):
            matrix[i][j]=abs(i-j)+1
    matrix=z_score(matrix,seq_len)

    return matrix


def process(filename):
    idx2rpd = {}
    fout = open(filename + '.rpd', 'wb')
    sentence_packs = json.load(open(filename))
    for sentence_pack in sentence_packs:
        rpd = relative_position_distance(sentence_pack['sentence'])
        idx2rpd[sentence_pack['id']] = rpd

    pickle.dump(idx2rpd, fout)
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