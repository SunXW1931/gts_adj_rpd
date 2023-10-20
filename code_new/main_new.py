#coding utf-8

import json, os
import random
import argparse

import torch
import torch.nn.functional as F
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

from data_new_2 import load_data_instances, DataIterator
from model_new import MultiInferRNNModel
import utils_new

from transformers import AdamW


def train(args):

    word2index = json.load(open(args.prefix + 'doubleembedding/word_idx.json'))
    general_embedding = np.load(args.prefix +'doubleembedding/gen.vec.npy')
    general_embedding = torch.from_numpy(general_embedding)
    domain_embedding = np.load(args.prefix +'doubleembedding/'+args.dataset+'_emb.vec.npy')
    domain_embedding = torch.from_numpy(domain_embedding)

    train_sentence_packs = json.load(open(args.prefix + args.dataset + '/train.json'))
    random.shuffle(train_sentence_packs)
    dev_sentence_packs = json.load(open(args.prefix + args.dataset + '/dev.json'))

    ftrain = args.prefix + args.dataset + '/train.json'
    fdev = args.prefix + args.dataset + '/dev.json'

    instances_train = load_data_instances(train_sentence_packs, word2index, args, ftrain)
    instances_dev = load_data_instances(dev_sentence_packs, word2index, args, fdev)

    random.shuffle(instances_train)

    trainset = DataIterator(instances_train, args)
    devset = DataIterator(instances_dev, args)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if args.model == 'bilstm':
        model = MultiInferRNNModel(general_embedding, domain_embedding, args).to(args.device)

    # optimizer = get_bert_optimizer(model, args)
    parameters = list(model.parameters())
    parameters = filter(lambda x: x.requires_grad, parameters)
    # optimizer = torch.optim.Adam(parameters, lr=args.lr)  # 改变优化器
    optimizer=torch.optim.AdamW(parameters,lr=args.lr)

    # label = ['N', 'B-A', 'I-A', 'A', 'B-O', 'I-O', 'O', 'negative', 'neutral', 'positive']
    weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).float().cuda()
    # weight = torch.tensor([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 1.95, 1.95, 1.95]).float().cuda()

    best_joint_f1 = 0
    best_joint_epoch = 0

    # loss函数绘制
    train_loss=[]
    for i in range(args.epochs):
        print('Epoch:{}'.format(i))
        for j in trange(trainset.batch_count):
            # 原始
            # _, sentence_tokens, sentence_poses, sentence_adjs, _, _, _, lengths, masks, aspect_tags, _, tags = trainset.get_batch(j)
            # predictions = model(sentence_tokens, sentence_poses, sentence_adjs, lengths, masks)

            # 改动：添加sentic_sentic_adjs
            # _, sentence_tokens, sentence_poses, _, sentence_sentic_adjs, _, _, lengths, masks, aspect_tags, _, tags = trainset.get_batch(j)
            # predictions = model(sentence_tokens, sentence_poses, sentence_sentic_adjs, lengths, masks)

            # 改动：添加sentic_sentic_adjs, sentence_rpds
            _, sentence_tokens, sentence_poses, _, sentence_sentic_adjs, _, sentence_rp, lengths, masks, aspect_tags, _, tags = trainset.get_batch(j)
            predictions = model(sentence_tokens, sentence_poses, sentence_sentic_adjs,  sentence_rp, lengths, masks)

            loss = 0.
            tags_flatten = tags[:, :lengths[0], :lengths[0]].reshape([-1])
            for k in range(len(predictions)):
                prediction_flatten = predictions[k].reshape([-1, predictions[k].shape[3]])
                loss = loss + F.cross_entropy(prediction_flatten, tags_flatten, weight=weight, ignore_index=-1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_data=loss.data.detach().cpu().numpy().squeeze()
        train_loss.append(loss_data.tolist())
        print('loss:', loss)

        joint_precision, joint_recall, joint_f1 = eval(model, devset, args)

        if joint_f1 > best_joint_f1:
            model_path = args.model_dir + args.model + args.task + '.pt'
            torch.save(model, model_path)
            best_joint_f1 = joint_f1
            best_joint_epoch = i
    print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(best_joint_epoch, args.task, best_joint_f1))

    # 绘制loss曲线
    with open("/media/a303/C07EC7757EC762B0/document/sxw/sxw/DGEIAN/DGEIAN-gts+adj+rpd/loss_curve/train_loss.txt",'w') as train_los:
        train_los.write(str(train_loss))


def eval(model, dataset, args):
    model.eval()
    with torch.no_grad():
        predictions=[]
        labels=[]
        all_ids = []
        all_lengths = []
        for i in range(dataset.batch_count):
            # 改动：sentence_adj
            # sentence_ids, sentence_tokens, sentence_poses, _, sentence_sentic_adjs, _, _, lengths, mask, aspect_tags, _, tags = dataset.get_batch(i)
            # prediction = model.forward(sentence_tokens, sentence_poses, sentence_sentic_adjs, lengths,mask)
            # 改动：sentence_adj+sentence_rpd
            sentence_ids, sentence_tokens, sentence_poses, _,sentence_sentic_adjs, _, sentence_rp, lengths, mask, aspect_tags, _, tags=dataset.get_batch(i)
            prediction = model.forward(sentence_tokens, sentence_poses, sentence_sentic_adjs,sentence_rp, lengths, mask)
            prediction = prediction[-1]
            prediction = torch.argmax(prediction, dim=3)
            prediction_padded = torch.zeros(prediction.shape[0], args.max_sequence_len, args.max_sequence_len)
            prediction_padded[:, :prediction.shape[1], :prediction.shape[1]] = prediction
            predictions.append(prediction_padded)

            all_ids.extend(sentence_ids)
            labels.append(tags)
            all_lengths.append(lengths)

        predictions = torch.cat(predictions,dim=0).cpu().tolist()
        labels = torch.cat(labels,dim=0).cpu().tolist()
        all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()
        precision, recall, f1 = utils_new.score_uniontags(args, predictions, labels, all_lengths, ignore_index=-1)

        aspect_results = utils_new.score_aspect(predictions, labels, all_lengths, ignore_index=-1)
        opinion_results = utils_new.score_opinion(predictions, labels, all_lengths, ignore_index=-1)
        print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1], aspect_results[2]))
        print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1], opinion_results[2]))
        print(args.task+'\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))


    model.train()
    return precision, recall, f1


def test(args):
    print("Evaluation on testset:")
    model_path = args.model_dir + args.model + args.task + '.pt'
    model = torch.load(model_path).to(args.device)
    model.eval()

    word2index = json.load(open(args.prefix + 'doubleembedding/word_idx.json'))
    sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
    ftest = args.prefix + args.dataset + '/test.json'
    instances = load_data_instances(sentence_packs, word2index, args, ftest)
    testset = DataIterator(instances, args)
    eval(model, testset, args)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="../data/",
                        help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default="savemodel/",
                        help='model path prefix')
    parser.add_argument('--task', type=str, default="triplet",
                        help='triplet')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--model', type=str, default="bilstm", choices=["bilstm"],
                        help='bilstm')
    parser.add_argument('--dataset', type=str, default="res15",
                        help='dataset')
    parser.add_argument('--max_sequence_len', type=int, default=100,
                        help='max length of a sentence')

    parser.add_argument('--pos_dim', type=int, default="100",
                        help='dimension of pos')

    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')

    parser.add_argument('--lstm_dim', type=int, default=150,
                        help='dimension of lstm cell')
    parser.add_argument('--cnn_dim', type=int, default=256,
                        help='dimension of cnn')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=600,
                        help='training epoch number')
    parser.add_argument('--class_num', type=int, default=10,  # 标签数量()
                        help='label number')
    parser.add_argument('--max_relative_position',type=int,default=4,  # 相对位置距离最大值
                        help='max relative position')
    # parser.add_argument('--seed', type=int, default=1000)

    args = parser.parse_args()

    # if args.seed is not None:
    #     random.seed(args.seed)
    #     np.random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     torch.cuda.manual_seed(args.seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    if args.mode == 'train':
        train(args)
        test(args)
    else:
        test(args)
    torch.cuda.empty_cache()
