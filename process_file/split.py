#encoding:utf-8
import codecs

split_filename = '/Users/yunfan/PycharmProjects/NLP_Learning/HAN/data/cnews.val.txt'  # validation data 验证集，当通过训练集训练出多个模型后，为了能找出效果最佳的模型，使用各个模型对验证集数据进行预测，并记录模型准确率。选出效果最佳的模型所对应的参数，即用来调整模型参数。
write_filename = '/Users/yunfan/PycharmProjects/NLP_Learning/HAN/data/test.val.txt'
i = 0
word = []
with codecs.open(split_filename,'r',encoding='utf-8') as fr:
    for line in fr:
        try:
            line=line.rstrip()
            assert len(line.split('\t')) == 2
            word.append(line)
            i=i+1
            if i > 200:
                break
        except:
            pass
with codecs.open(write_filename,'w',encoding='utf-8') as fw:
    fw.write('\n'.join(word))