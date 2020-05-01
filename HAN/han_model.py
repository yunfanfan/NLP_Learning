# -*- coding: utf-8 -*-
import tensorflow as tf

class TextConfig(object):


    embedding_dim = 100      #dimension of word embedding
    pre_trianing=None        #use vector_char trained by word2vec
    vocab_size = 6000 # 整篇文章有多少个词

    seq_length=7
    num_classes=10
    max_sent_in_doc = 30
    max_word_in_sent = 8

    keep_prob=0.5          #droppout
    learning_rate= 1e-3    #learning rate
    lr_decay= 0.9          #learning rate decay
    grad_clip= 5.0         #gradient clipping threshold

    num_layers= 1           #the number of layer
    hidden_dim = 128        #the number of hidden units
    attention_size = 100    #the size of attention layer

    num_epochs=10          #epochs 1个epoch表示过了1遍训练集中的所有样本;
    batch_size= 64         #batch_size 1次迭代所使用的样本量；
    print_per_batch =100   #print result

    train_filename = './data/cnews.train.txt'  # train data 训练集，用来拟合模型，通过设置分类器的参数，训练分类模型。
    test_filename = './data/cnews.test.txt'  # test data 测试集，当已经确定模型参数后，使用测试集进行模型性能评价。
    val_filename = './data/cnews.val.txt'  # validation data 验证集，当通过训练集训练出多个模型后，为了能找出效果最佳的模型，使用各个模型对验证集数据进行预测，并记录模型准确率。选出效果最佳的模型所对应的参数，即用来调整模型参数。
    vocab_filename = './data/vocab.txt'  # vocabulary
    vector_word_filename = './data/vector_word.txt'  # vector_word trained by word2vec
    vector_word_npz = './data/vector_word.npz'  # save vector_word to numpy file
    test = './data/test.txt'

class TextHAN(object):
    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, self.config.max_sent_in_doc, self.config.max_word_in_sent], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths") #?
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.max_sentence_num = tf.placeholder(tf.int32, name='max_sentence_num')
        self.max_sentence_length = tf.placeholder(tf.int32, name='max_sentence_length')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')

        self.han()

    def han(self):

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # self.embedding = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embedding_dim], -1.0, 1.0), trainable=False,name='W')
            # 获取已存在的变量（要求不仅名字，而且初始化方法等各个参数都一样），如果不存在，就新建一个。
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_dim],
                                             initializer=tf.constant_initializer(self.config.pre_trianing))
            # 选取一个张量里面索引对应的元素。
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        # Sent2vec
        with tf.name_scope("sent2vec"):
            # GRU的输入tensor是[batch_size, max_time, ...].在构造句子向量时max_time应该是每个句子的长度，所以这里将
            # batch_size * sent_in_doc当做是batch_size.这样一来，每个GRU的cell处理的都是一个单词的词向量
            # 并最终将一句话中的所有单词的词向量融合（Attention）在一起形成句子向量

            # shape为[batch_size*sent_in_doc, word_in_sent, embedding_size]
            word_embedded = tf.reshape(embedding_inputs, [-1, self.max_sentence_length, self.config.embedding_dim])
            # shape为[batch_size*sent_in_doce, word_in_sent, hidden_size*2]
            word_encoded = BidirectionalGRUEncoder(word_embedded, name='word_encoder')
            # shape为[batch_size*sent_in_doc, hidden_size*2]
            sent_vec = self.config.AttentionLayer(word_encoded, name='word_attention')

        #原理与sent2vec一样，根据文档中所有句子的向量构成一个文档向量
        with tf.name_scope("doc2vec"):
            sent_vec = tf.reshape(sent_vec, [-1, self.max_sentence_num, self.config.hidden_size*2])
            #shape为[batch_size, sent_in_doc, hidden_size*2]
            doc_encoded = self.config.BidirectionalGRUEncoder(sent_vec, name='sent_encoder')
            #shape为[batch_szie, hidden_szie*2]
            doc_vec = self.config.AttentionLayer(doc_encoded, name='sent_attention')

        # Add dropout
        with tf.name_scope('dropout'):
            # attention_output shape: (batch_size, hidden_size)
            self.final_output = tf.nn.dropout(doc_vec, self.keep_prob)

        #最终的输出层，是一个全连接层
        with tf.name_scope('doc_classification'):
            out = tf.contrib.layers.fully_connected(inputs=self.final_output, num_outputs=self.config.num_classes, activation_fn=None)
            self.logits = out

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                      logits=self.logits,
                                                                      name='loss'))
        # Create optimizer
        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.grad_clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        with tf.name_scope('accuracy'):
            self.y_pred_cls = tf.argmax(self.logits, axis=1, name='predict')
            label = tf.argmax(self.input_y, axis=1, name='label')
            acc = tf.reduce_mean(tf.cast(tf.equal(self.y_pred_cls, label), tf.float32))

        def BidirectionalGRUEncoder(self, inputs, name):
            # 双向GRU的编码层，将一句话中的所有单词或者一个文档中的所有句子向量进行编码得到一个 2×hidden_size的输出向量，然后在经过Attention层，将所有的单词或句子的输出向量加权得到一个最终的句子/文档向量。
            # 输入inputs的shape是[batch_size, max_time, voc_size]
            with tf.variable_scope(name):
                # Define Forward GRU Cell
                GRU_cell_fw = tf.contrib.rnn.GRUCell(self.hidden_size)
                GRU_cell_fw = tf.contrib.rnn.DropoutWrapper(GRU_cell_fw, output_keep_prob=self.keep_prob)
                # Define Backward GRU Cell
                GRU_cell_bw = tf.contrib.rnn.GRUCell(self.hidden_size)
                GRU_cell_bw = tf.contrib.rnn.DropoutWrapper(GRU_cell_bw, output_keep_prob=self.keep_prob)
                # fw_outputs和bw_outputs的size都是[batch_size, max_time, hidden_size]
                ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                     cell_bw=GRU_cell_bw,
                                                                                     inputs=inputs,
                                                                                     sequence_length=self.sequence_lengths,
                                                                                     dtype=tf.float32)
                # outputs的size是[batch_size, max_time, hidden_size*2]
                outputs = tf.concat((fw_outputs, bw_outputs), 2)
                return outputs

        def AttentionLayer(self, inputs, name):
            # Attention Layer
            with tf.name_scope('attention'):
                # inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
                # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
                # 因为使用双向GRU，所以其长度为2×hidden_szie
                u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')
                # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
                h = tf.contrib.layers.fully_connected(out, self.hidden_size * 2, activation_fn=tf.nn.tanh)
                # shape为[batch_size, max_time, 1]
                alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
                # reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
                atten_output = tf.reduce_sum(tf.multiply(out, alpha), axis=1)

                return atten_output
