import math
import pickle

import numpy as np
import torch
import spacy

from collections import OrderedDict,defaultdict

nlp = spacy.load('en_core_web_sm')

# sentiment2id = {'negative': 3, 'neutral': 4, 'positive': 5}
# 改动：在GTS中增加'B-A','I-A','B-O','I-O'
label=['N','B-A','I-A','A','B-O','I-O','O','negative','neutral','positive']

label2id, id2label = OrderedDict(), OrderedDict()
for i, v in enumerate(label):
    label2id[v] = i
    id2label[i] = v


def get_spans(tags):
    '''for BIO tag'''
    tags = tags.strip().split()
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


def getpos(tags,args):
    pos = torch.zeros(args.max_sequence_len).long()
    for i, tags in enumerate(tags):
        word, tag = tags
        if tag.startswith('NN'):  # 名词单数
            pos[i] = 1
        elif tag.startswith('VB'):  # 动词原形
            pos[i] = 2
        elif tag.startswith('JJ'):  # 形容词
            pos[i] = 3
        elif tag.startswith('RB'):  # 副词
            pos[i] = 4
        else:
            pos[i] = 0
    return pos


class Instance(object):
    def __init__(self, sentence_pack, word2index, args, fname):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']
        self.sentence_tokens = torch.zeros(args.max_sequence_len).long()
        words = self.sentence.split()
        self.length = len(words)
        for i, w in enumerate(words):
            word = w
            if word in word2index:
                self.sentence_tokens[i] = word2index[word]
            else:
                self.sentence_tokens[i] = word2index['<unk>']

        # POS编码
        f = open(fname + '.pos', 'rb')
        idx2pos = pickle.load(f)
        f.close()
        for key in idx2pos.keys():
            if key == self.id:
                self.sentence_pos = getpos(idx2pos[key],args)
                break

        # 邻接矩阵
        self.sentence_adj = torch.zeros(self.length, self.length).long()
        fin = open(fname + '.graph', 'rb')
        idx2graph = pickle.load(fin)
        fin.close()
        for key in idx2graph.keys():
            if key == self.id:
                self.sentence_adj = idx2graph[key]
                break

        # 改动：添加情感信息的邻接矩阵
        self.sentence_sentic_adj=torch.zeros(self.length,self.length).long()
        fin=open(fname+'.sentic_dg','rb')
        idx2sentic_adj=pickle.load(fin)
        fin.close()
        for key in idx2sentic_adj.keys():
            if key==self.id:
                self.sentence_sentic_adj=idx2sentic_adj[key]
                break

        # 改动：添加情感信息
        self.sentence_sentic=torch.zeros(self.length,self.length).long()
        fin=open(fname+'.sentic','rb')
        idx2sentic=pickle.load(fin)
        fin.close()
        for key in idx2sentic.keys():
            if key==self.id:
                self.sentence_sentic=idx2sentic[key]
                break

        self.aspect_tags = torch.zeros(args.max_sequence_len).long()
        self.opinion_tags = torch.zeros(args.max_sequence_len).long()
        self.aspect_tags[self.length:] = -1
        self.opinion_tags[self.length:] = -1
        self.tags = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        self.tags[:, :] = -1

        for i in range(self.length):
            for j in range(i, self.length):
                self.tags[i][j] = 0

        for pair in sentence_pack['triples']:
            aspect = pair['target_tags']
            opinion = pair['opinion_tags']
            aspect_span = get_spans(aspect)
            opinion_span = get_spans(opinion)

            # 改动：添加aspect_tags
            for l, r in aspect_span:
                for i in range(l, r + 1):
                    self.aspect_tags[i] = 1 if i == l else 2  # 第一个aspect是1，后面都是2
                    for j in range(i, r + 1):
                        if j == l:
                            self.tags[i][j] = label2id['B-A']
                        elif j == i:
                            self.tags[i][j] = label2id['I-A']
                        else:
                            self.tags[i][j] = label2id['A']

            # 改动：添加opinion_tags
            for l, r in opinion_span:
                for i in range(l, r + 1):
                    self.opinion_tags[i] = 1 if i == l else 2  # 第一个aspect是1，后面都是2
                    for j in range(i, r + 1):
                        if j == l:
                            self.tags[i][j] = label2id['B-O']
                        elif j == i:
                            self.tags[i][j] = label2id['I-O']
                        else:
                            self.tags[i][j] = label2id['O']

            for al, ar in aspect_span:
                for pl, pr in opinion_span:
                    for i in range(al, ar+1):
                        for j in range(pl, pr+1):
                            if args.task == 'pair':
                                if i > j: self.tags[j][i] = 7
                                else: self.tags[i][j] = 7
                            elif args.task == 'triplet':
                                if i > j: self.tags[j][i] = label2id[pair['sentiment']]
                                else: self.tags[i][j] = label2id[pair['sentiment']]

        # tag1=np.array(self.tags)
        # print(tag1)

        '''generate mask of the sentence'''
        self.mask = torch.zeros(args.max_sequence_len)
        self.mask[:self.length] = 1


def load_data_instances(sentence_packs, word2index, args, fname):
    instances = list()
    for sentence_pack in sentence_packs:
        instances.append(Instance(sentence_pack, word2index, args, fname))
    return instances


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)  # 向上取整

    def get_batch(self, index):
        sentence_ids = []
        sentence_tokens = []
        sentence_poses = []
        sentence_adjs = []
        sentence_sentic_adjs=[]
        sentence_sentics=[]
        lengths = []
        masks = []
        aspect_tags = []
        opinion_tags = []
        tags = []

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):

            sentence_ids.append(self.instances[i].id)
            sentence_tokens.append(self.instances[i].sentence_tokens)
            lengths.append(self.instances[i].length)
            masks.append(self.instances[i].mask)
            aspect_tags.append(self.instances[i].aspect_tags)
            opinion_tags.append(self.instances[i].opinion_tags)
            tags.append(self.instances[i].tags)

        max_len = max(lengths)

        # 添加pos
        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            pos = self.instances[i].sentence_pos
            sentence_poses.append(pos)

        # 添加邻接矩阵
        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            len_s = self.instances[i].length
            a = self.instances[i].sentence_adj
            adj = np.pad(a, ((0,max_len-len_s),(0,max_len-len_s)), 'constant')
            adj = torch.from_numpy(adj)
            sentence_adjs.append(adj)

        # 改动：添加sentic_adj
        for i in range(index*self.args.batch_size,
                       min((index+1)*self.args.batch_size,len(self.instances))):
            len_s=self.instances[i].length
            s_a=self.instances[i].sentence_sentic_adj
            sentic_adj=np.pad(s_a, ((0,max_len-len_s),(0,max_len-len_s)), 'constant')
            sentic_adj=torch.from_numpy(sentic_adj)
            sentence_sentic_adjs.append(sentic_adj)

        # 改动：添加sentic
        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            len_s = self.instances[i].length
            s = self.instances[i].sentence_sentic
            sentic = np.pad(s, ((0, max_len - len_s), (0, max_len - len_s)), 'constant')
            sentic = torch.from_numpy(sentic)
            sentence_sentics.append(sentic)

        # 改动：添加相对位置距离
        length=max_len
        max_rp=self.args.max_relative_position
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(0).expand(length, length)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        for i in range(length):
            for j in range(length):
                if distance_mat[i][j] < -max_rp:
                    distance_mat[i][j] = -max_rp
                elif distance_mat[i][j] > max_rp:
                    distance_mat[i][j] = max_rp
        sentence_rp = distance_mat + max_rp

        # 改动：添加相对位置距离1
        # for i in range(index * self.args.batch_size,
        #                min((index + 1) * self.args.batch_size, len(self.instances))):
        #     len_s = self.instances[i].length
        #     p = self.instances[i].sentence_rpd
        #     position = np.pad(p, ((0, max_len - len_s), (0, max_len - len_s)), 'constant')
        #     position = torch.from_numpy(position)
        #     sentence_rpds.append(position)

        indexes = list(range(len(sentence_tokens)))
        indexes = sorted(indexes, key=lambda x: lengths[x], reverse=True)  # 排序 reverse=True为降序

        sentence_ids = [sentence_ids[i] for i in indexes]
        sentence_tokens = torch.stack(sentence_tokens).to(self.args.device)[indexes]
        sentence_poses = torch.stack(sentence_poses).to(self.args.device)[indexes]
        sentence_adjs = torch.stack(sentence_adjs).to(self.args.device)[indexes]
        sentence_sentic_adjs = torch.stack(sentence_sentic_adjs).to(self.args.device)[indexes]
        sentence_sentics = torch.stack(sentence_sentics).to(self.args.device)[indexes]
        lengths = torch.tensor(lengths).to(self.args.device)[indexes]
        masks = torch.stack(masks).to(self.args.device)[indexes]
        aspect_tags = torch.stack(aspect_tags).to(self.args.device)[indexes]
        opinion_tags = torch.stack(opinion_tags).to(self.args.device)[indexes]
        tags = torch.stack(tags).to(self.args.device)[indexes]
        sentence_rp=sentence_rp.to(self.args.device)

        return sentence_ids, sentence_tokens, sentence_poses, sentence_adjs, sentence_sentic_adjs, sentence_sentics, sentence_rp, lengths, masks, aspect_tags, opinion_tags, tags
