# encoding:utf-8
from collections import Counter
import tensorflow.keras as kr
import numpy as np
import codecs
import re
import jieba


def read_file(filename):
    """
    Args:
        filename:trian_filename,test_filename,val_filename
    Returns:
        two list where the first is lables and the second is contents cut by jieba

    """
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation

    with codecs.open('./data/stopwords.txt', 'r', encoding='utf-8') as f:
        # 列表解析
        stopwords = [line.strip() for line in f.readlines()]

    contents, labels = [], []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                #  rstrip() 删除 string 字符串末尾的指定字符（默认为空格）
                line = line.rstrip()
                assert len(line.split('\t')) == 2
                label, content = line.split('\t')
                labels.append(label)
                blocks = re_han.split(content)
                # word[]存的是分词后的内容
                word = []
                for blk in blocks:
                    # re.match 尝试从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match()就返回none。
                    if re_han.match(blk):
                        # lcut()返回列表，cut()返回迭代器
                        seglist = jieba.lcut(blk)
                        # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
                        # 去掉停用词，并且把jieba分词后的结果给word[]列表
                        word.extend([w for w in seglist if w not in stopwords])
                contents.append(word)
            except:
                pass
    return labels, contents


def build_vocab(filenames, vocab_dir, vocab_size=8000):
    """
    Args:
        filename:trian_filename,test_filename,val_filename
        vocab_dir:path of vocab_filename
        vocab_size:number of vocabulary
    Returns:
        writting vocab to vocab_filename
        输出一个按词频排列的词汇表

    """
    all_data = []
    for filename in filenames:
        _, data_train = read_file(filename)
        for content in data_train:
            all_data.extend(content)
    # counter作用就是在一个数组内，遍历所有元素，将元素出现的次数记下来
    counter = Counter(all_data)
    # Counter(a).most_common(2)可以打印出数组中出现次数最多的元素。
    # 参数2表示的含义是：输出几个出现次数最多的元素。
    count_pairs = counter.most_common(vocab_size - 1)
    # *parameter是用来接受任意多个参数并将其放在一个元组中
    # 在 Python 3.x 中为了减少内存，zip() 返回的是一个对象。如需展示列表，需手动 list() 转换。
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)

    with codecs.open(vocab_dir, 'w', encoding='utf-8') as f:
        # join 返回通过指定字符连接序列中元素后生成的新字符串。
        f.write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """
    Args:
        filename:path of vocab_filename
    Returns:
        words: a list of vocab
        word_to_id: a dict of word to id

    """
    words = codecs.open(vocab_dir, 'r', encoding='utf-8').read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """
    Args:
        None
    Returns:
        categories: a list of label
        cat_to_id: a dict of label to id

    """
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def process_file(filename, word_to_id, cat_to_id, max_length=200):
    """
    Args:
        filename:train_filename or test_filename or val_filename
        word_to_id:get from def read_vocab()
        cat_to_id:get from def read_category()
        max_length:allow max length of sentence
    Returns:
        x_pad: sequence data from  preprocessing sentence
        y_pad: sequence data from preprocessing label

    """
    labels, contents = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        # word_to_id词在dic中的id位置，把文中的一个一个的词转换成我们词表中对应的数字
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    # pad_sequences:将多个序列截断或补齐为相同长度。
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')
    # 将整型标签转为onehot
    y_pad = kr.utils.to_categorical(label_id)
    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """
    Args:
        x: x_pad get from def process_file()
        y:y_pad get from def process_file()
    Yield:
        input_x,input_y by batch size

    """

    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def export_word2vec_vectors(vocab, word2vec_dir, trimmed_filename):
    """
    Args:
        vocab: word_to_id
        word2vec_dir:file path of have trained word vector by word2vec
        trimmed_filename:file path of changing word_vector to numpy file
    Returns:
        save vocab_vector to numpy file

    """
    file_r = codecs.open(word2vec_dir, 'r', encoding='utf-8')

    line = file_r.readline()  # 读第一行数据"412955 100"
    voc_size, vec_dim = map(int, line.split(' '))  # voc_size = 412955, vec_dim = 100
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            # 将结构数据转化为ndarray，不会占用新的内存
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()
    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_training_word2vec_vectors(filename):
    """
    Args:
        filename:numpy file
    Returns:
        data["embeddings"]: a matrix of vocab vector
    """
    with np.load(filename) as data:
        return data["embeddings"]


def get_sequence_length(x_batch):
    """
    Args:
        x_batch:a batch of input_data
    Returns:
        sequence_lenghts: a list of acutal length of  every senuence_data in input_data
    """
    sequence_lengths = []
    for x in x_batch:
        actual_length = np.sum(np.sign(x))
        sequence_lengths.append(actual_length)
    return sequence_lengths
