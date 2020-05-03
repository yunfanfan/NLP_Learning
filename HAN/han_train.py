#encoding:utf-8
from __future__ import print_function

from han_model import *
from han_loader import *
import os
import time

def feed_data(x_batch, y_batch, keep_prob):
    sequence_lengths = get_sequence_length(x_batch)
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob:keep_prob,
        model.sequence_lengths: sequence_lengths,
        model.max_sentence_length:8,
        model.max_sentence_num:30
    }
    return feed_dict

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
    #x_train, y_train = han_process_file(config.train_filename, word_to_id, cat_to_id, config.seq_length)
    x_train, y_train = han_process_file(config.test,word_to_id,cat_to_id,config.max_sent_in_doc,config.max_word_in_sent)
    #x_val, y_val = han_process_file(config.val_filename, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = han_process_file(config.test,word_to_id,cat_to_id,config.max_sent_in_doc,config.max_word_in_sent)
    print("Time cost: %.3f seconds...\n" % (time.time() - start_time))

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)


    print('Training and evaluating...')
    best_val_accuracy = 0
    last_improved = 0  # record global_step at best_val_accuracy
    require_improvement = 1000  # break training if not having improvement over 1000 iter
    flag=False

    for epoch in range(config.num_epochs):
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        start = time.time()
        print('Epoch:', epoch + 1)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.keep_prob)
            _, global_step, train_summaries, train_loss, train_accuracy = session.run([model.optim, model.global_step,
                                                                                       merged_summary, model.loss,
                                                                                       model.acc], feed_dict=feed_dict)
            if global_step % config.print_per_batch == 0:
                end = time.time()
                feed_dict = feed_data(x_val, y_val, 1.0)
                # 使用验证集的数据
                val_summaries, val_loss, val_accuracy = session.run([merged_summary, model.loss, model.acc],
                                                                    feed_dict=feed_dict)
                writer.add_summary(val_summaries, global_step)
                # If improved, save the model
                if val_accuracy > best_val_accuracy:
                    saver.save(session, save_path)
                    best_val_accuracy = val_accuracy
                    last_improved=global_step
                    improved_str = '*'
                else:
                    improved_str = ''
                print("step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, val accuracy: {:.3f},training speed: {:.3f}sec/batch {}\n".format(
                    global_step, train_loss, train_accuracy, val_loss, val_accuracy,
                    (end - start) / config.print_per_batch,improved_str))
                start = time.time()

            if global_step - last_improved > require_improvement:
                print("No optimization over 1000 steps, stop training")
                flag = True
                break
        if flag:
            break
        config.learning_rate *= config.lr_decay


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

    model = TextHAN(config)
    train()
