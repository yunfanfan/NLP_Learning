# -*- coding: utf-8 -*-
import tensorflow as tf

class TextConfig(object):


    vocab_size = 6000 # 整篇文章有多少个词

    seq_length=7
    num_classes=10


    train_filename = './data/cnews.train.txt'  # train data 训练集，用来拟合模型，通过设置分类器的参数，训练分类模型。
    test_filename = './data/cnews.test.txt'  # test data 测试集，当已经确定模型参数后，使用测试集进行模型性能评价。
    val_filename = './data/cnews.val.txt'  # validation data 验证集，当通过训练集训练出多个模型后，为了能找出效果最佳的模型，使用各个模型对验证集数据进行预测，并记录模型准确率。选出效果最佳的模型所对应的参数，即用来调整模型参数。
    vocab_filename = './data/vocab.txt'  # vocabulary
    vector_word_filename = './data/vector_word.txt'  # vector_word trained by word2vec
    vector_word_npz = './data/vector_word.npz'  # save vector_word to numpy file
    test = './data/test.txt'

#class TextHAN(object):
