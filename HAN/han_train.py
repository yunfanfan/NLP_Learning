#encoding:utf-8
from __future__ import print_function
from han_model import *
from han_loader import *
import os
import time

def train():
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = './tensorboard/texthan'
    save_dir = './checkpoints/texthan'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 拼接路径
    save_path = os.path.join(save_dir, 'best_validation')

    print("Loading training and validation data...")
    start_time = time.time()
    #x_train, y_train = process_file(config.train_filename, word_to_id, cat_to_id, config.seq_length)
    x_train, y_train = han_process_file(config.test,word_to_id,cat_to_id,config.seq_length)
    #x_val, y_val = process_file(config.val_filename, word_to_id, cat_to_id, config.seq_length)
    print("Time cost: %.3f seconds...\n" % (time.time() - start_time))


if __name__ == '__main__':
    print('Configuring HAN model...')
    config = TextConfig()
    #filenames = [config.train_filename, config.test_filename, config.val_filename]
    filenames = [config.test]
    if not os.path.exists(config.vocab_filename): # 如果没有分好的词 vocal.txt，就执行build_vocab
        build_vocab(filenames, config.vocab_filename, config.vocab_size)

    #read vocab and categories
    categories,cat_to_id = read_category()
    words,word_to_id = read_vocab(config.vocab_filename)
    config.vocab_size = len(words)

    # trans vector file to numpy file
    if not os.path.exists(config.vector_word_npz):
        export_word2vec_vectors(word_to_id, config.vector_word_filename, config.vector_word_npz)
    config.pre_trianing = get_training_word2vec_vectors(config.vector_word_npz)

    #model = TextHAN(config)
    train()
