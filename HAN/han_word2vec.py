#encoding:utf-8
import logging
import time
import codecs
import sys
import re
import jieba
from gensim.models import word2vec
from han_model import TextConfig

re_han= re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)") # the method of cutting text by punctuation


class Get_Sentences(object):
    '''

    Args:
         filenames: a list of train_filename,test_filename,val_filename
    Yield:
        word:a list of word cut by jieba

    '''

    def __init__(self,filenames):
        self.filenames= filenames

    def __iter__(self):
        for filename in self.filenames:
            # codecs.open可以打开由不同编码格式组成的文件
            # with 语句适用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的“清理”操作，释放资源
            with codecs.open(filename, 'r', encoding='utf-8') as f:
                # enumerate()将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
                for _,line in enumerate(f):
                    try:
                        # strip()去除首尾空格
                        line=line.strip()
                        # "\t"制表符
                        line=line.split('\t')
                        # 如果不满足就直接返回错误
                        assert len(line)==2
                        blocks=re_han.split(line[1])
                        word=[]
                        for blk in blocks:
                            if re_han.match(blk):
                                word.extend(jieba.lcut(blk))
                        yield word
                    except:
                        pass

def train_word2vec(filenames):
    '''
    use word2vec train word vector
    argv:
        filenames: a list of train_filename,test_filename,val_filename
    return:
        save word vector to config.vector_word_filename

    '''
    # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）
    t1 = time.time()
    sentences = Get_Sentences(filenames)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=1, workers=12)
    model.wv.save_word2vec_format(config.vector_word_filename, binary=False)
    print('-------------------------------------------')
    print("Training word2vec model cost %.3f seconds...\n" % (time.time() - t1))

if __name__ == '__main__':
    config=TextConfig()
    #filenames=[config.train_filename,config.test_filename,config.val_filename]
    filenames=[config.test]
    train_word2vec(filenames)

