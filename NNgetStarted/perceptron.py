# -*- coding: utf-8 -*-
from functools import reduce


class Perceptron(object):
    def __init__(self, inputNum, activator):
        '''
        :param inputNum: 输入参数个数
        :param activator: 激活函数
        '''
        self.activator = activator
        # 权重向量初始化为0，_被称为丢弃变量，用完就不需要了
        self.weights = [0.0 for _ in range(inputNum)]
        # 偏置项初始化为0
        self.bias = 0.0

    def __str__(self):
        '''
        str方法用于打印类的实例对象时被调用，一般返回一个字符串
        :return: 学习到的权重、偏置项
        '''
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        '''
        :param input_vec: 输入向量
        :return: 感知器的计算结果
        '''
        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 最后利用reduce求和
        return self.activator(
            reduce(lambda a, b: a + b,
                   map(lambda x, w: x * w,
                       input_vec, self.weights)
                   , 0.0) + self.bias)
    def train(self, input_vecs, labels, iteration, rate):
        '''
        :param input_vecs: 一组向量
        :param labels: 与每个向量对应的label
        :param iteration: 训练轮数
        :param rate: 学习率
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        '''
        :return: 一次迭代，把所有训练数据过一遍
        '''
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = zip(input_vecs, labels)
        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        '''
        按照感知器规则更新权重
        '''
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        delta = label - output
        self.weights = map(
            lambda (x, w): w + rate * delta * x,
            zip(input_vec, self.weights))
        # 更新bias
        self.bias += rate * delta