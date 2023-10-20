import multiprocessing
import pickle
import numpy as np
import sklearn
from data_new_2 import label2id, id2label


def get_aspects(tags, length, ignore_index=-1):
    spans = []
    start, end = -1, -1
    for i in range(length):
        if tags[i][i] == ignore_index:
            continue
        label = id2label[tags[i][i]]
        if label == 'B-A':
            if start != -1:
                spans.append([start, end])
            start, end = i, i
        elif label == 'I-A':
            end = i
        else:
            if start != -1:
                spans.append([start, end])
                start, end = -1, -1
    if start != -1:
        spans.append([start, length-1])
    return spans


def get_opinions(tags, length, ignore_index=-1):
    spans = []
    start, end = -1, -1
    for i in range(length):
        if tags[i][i] == ignore_index:
            continue
        label = id2label[tags[i][i]]
        if label == 'B-O':
            if start != -1:
                spans.append([start, end])
            start, end = i, i
        elif label == 'I-O':
            end = i
        else:
            if start != -1:
                spans.append([start, end])
                start, end = -1, -1
    if start != -1:
        spans.append([start, length - 1])
    return spans

def score_aspect(predicted, golden, lengths, ignore_index=-1):
    assert len(predicted) == len(golden)
    golden_set = set()
    predict_set = set()
    for i in range(len(golden)):
        golden_spans = get_aspects(golden[i], lengths[i], ignore_index)
        for l, r in golden_spans:
            golden_set.add('-'.join([str(i), str(l), str(r)]))

        predict_spans = get_aspects(predicted[i], lengths[i], ignore_index)
        for l, r in predict_spans:
            predict_set.add('-'.join([str(i), str(l), str(r)]))

    correct_num = len(golden_set & predict_set)
    precision = correct_num / len(predict_set) if len(predict_set) > 0 else 0
    recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def score_opinion(predicted, golden, lengths, ignore_index=-1):
    assert len(predicted) == len(golden)
    golden_set = set()
    predict_set = set()
    for i in range(len(golden)):
        golden_spans = get_opinions(golden[i], lengths[i], ignore_index)
        for l, r in golden_spans:
            golden_set.add('-'.join([str(i), str(l), str(r)]))

        predict_spans = get_opinions(predicted[i], lengths[i], ignore_index)
        for l, r in predict_spans:
            predict_set.add('-'.join([str(i), str(l), str(r)]))

    correct_num = len(golden_set & predict_set)
    precision = correct_num / len(predict_set) if len(predict_set) > 0 else 0
    recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def find_pair(tags, aspect_spans, opinion_spans):
    pairs = []
    for al, ar in aspect_spans:
        for pl, pr in opinion_spans:
            tag_num = [0] * 4
            for i in range(al, ar + 1):
                for j in range(pl, pr + 1):
                    if al < pl:
                        tag_num[int(tags[i][j])] += 1
                    else:
                        tag_num[int(tags[j][i])] += 1
            if tag_num[3] == 0: continue
            sentiment = -1
            pairs.append([al, ar, pl, pr])
    return pairs

def find_triplet(tags, aspect_spans, opinion_spans):
    # label2id = {'N': 0, 'B-A': 1, 'I-A': 2, 'A': 3, 'B-O': 4, 'I-O': 5, 'O': 6, 'negative': 7, 'neutral': 8, 'positive': 9}
    triplets = []
    for al, ar in aspect_spans:
        for pl, pr in opinion_spans:
            tag_num = [0] * len(label2id)
            for i in range(al, ar + 1):
                for j in range(pl, pr + 1):
                    if al < pl:
                        tag_num[int(tags[i][j])] += 1
                    else:
                        tag_num[int(tags[j][i])] += 1

            if sum(tag_num[7:]) == 0: continue
            sentiment = -1
            if tag_num[9] >= tag_num[8] and tag_num[9] >= tag_num[7]:
                sentiment = 9
            elif tag_num[8] >= tag_num[7] and tag_num[8] >= tag_num[9]:
                sentiment = 8
            elif tag_num[7] >= tag_num[9] and tag_num[7] >= tag_num[8]:
                sentiment = 7
            if sentiment == -1:
                print('wrong!!!!!!!!!!!!!!!!!!!!')
                input()
            triplets.append([al, ar, pl, pr, sentiment])
    return triplets

def score_uniontags(args, predicted, golden, lengths, ignore_index=-1):
    assert len(predicted) == len(golden)
    golden_set = set()
    predicted_set = set()
    for i in range(len(golden)):
        golden_aspect_spans = get_aspects(golden[i], lengths[i], ignore_index)
        golden_opinion_spans = get_opinions(golden[i], lengths[i], ignore_index)
        if args.task == 'pair':
            golden_tuple = find_pair(golden[i], golden_aspect_spans, golden_opinion_spans)
        elif args.task == 'triplet':
            golden_tuple = find_triplet(golden[i], golden_aspect_spans, golden_opinion_spans)
        for pair in golden_tuple:
            golden_set.add(str(i) + '-'+ '-'.join(map(str, pair)))

        predicted_aspect_spans = get_aspects(predicted[i], lengths[i], ignore_index)
        predicted_opinion_spans = get_opinions(predicted[i], lengths[i], ignore_index)
        if args.task == 'pair':
            predicted_tuple = find_pair(predicted[i], predicted_aspect_spans, predicted_opinion_spans)
        elif args.task == 'triplet':
            predicted_tuple = find_triplet(predicted[i], predicted_aspect_spans, predicted_opinion_spans)
        for pair in predicted_tuple:
            predicted_set.add(str(i) + '-' + '-'.join(map(str, pair)))

    correct_num = len(golden_set & predicted_set)
    precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
    recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1
