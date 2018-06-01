import atexit
import json
import os
import pickle
import random
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper

from datasets import load_vte_dataset, ImageReader
from embeddings import load_glove, glove_embeddings_initializer
from utils import Progbar
from utils import batch
from utils import start_logger, stop_logger


def build_simple_vte_model_relu_phi(premise_input,
                                    hypothesis_input,
                                    img_features_input,
                                    dropout_input,
                                    num_tokens,
                                    num_labels,
                                    embeddings,
                                    embeddings_size,
                                    train_embeddings,
                                    rnn_hidden_size,
                                    multimodal_fusion_hidden_size,
                                    classification_hidden_size):
    premise_length = tf.cast(
        tf.reduce_sum(
            tf.cast(tf.not_equal(premise_input, tf.zeros_like(premise_input, dtype=tf.int32)), tf.int64),
            1
        ),
        tf.int32
    )
    hypothesis_length = tf.cast(
        tf.reduce_sum(
            tf.cast(tf.not_equal(hypothesis_input, tf.zeros_like(hypothesis_input, dtype=tf.int32)), tf.int64),
            1
        ),
        tf.int32
    )
    if embeddings is not None:
        embedding_matrix = tf.get_variable(
            "embedding_matrix",
            shape=(num_tokens, embeddings_size),
            initializer=glove_embeddings_initializer(embeddings),
            trainable=train_embeddings
        )
        print("Loaded GloVe embeddings!")
    else:
        embedding_matrix = tf.get_variable(
            "embedding_matrix",
            shape=(num_tokens, embeddings_size),
            initializer=tf.random_normal_initializer(stddev=0.05),
            trainable=train_embeddings
        )
    premise_embeddings = tf.nn.embedding_lookup(embedding_matrix, premise_input)
    hypothesis_embeddings = tf.nn.embedding_lookup(embedding_matrix, hypothesis_input)
    lstm_cell = DropoutWrapper(
        tf.nn.rnn_cell.LSTMCell(rnn_hidden_size),
        input_keep_prob=dropout_input,
        output_keep_prob=dropout_input
    )
    premise_outputs, premise_final_states = tf.nn.dynamic_rnn(
        cell=lstm_cell,
        inputs=premise_embeddings,
        sequence_length=premise_length,
        dtype=tf.float32
    )
    hypothesis_outputs, hypothesis_final_states = tf.nn.dynamic_rnn(
        cell=lstm_cell,
        inputs=hypothesis_embeddings,
        sequence_length=hypothesis_length,
        dtype=tf.float32
    )
    normalized_img_features = tf.nn.l2_normalize(img_features_input, dim=1)
    img_hidden_layer = tf.nn.dropout(
        tf.contrib.layers.fully_connected(normalized_img_features, multimodal_fusion_hidden_size),
        keep_prob=dropout_input
    )
    hypothesis_hidden_layer = tf.nn.dropout(
        tf.contrib.layers.fully_connected(hypothesis_final_states.h, multimodal_fusion_hidden_size),
        keep_prob=dropout_input
    )
    hypothesis_img_multimodal_fusion = tf.multiply(hypothesis_hidden_layer, img_hidden_layer)
    final_concatenation = tf.concat([premise_final_states.h, hypothesis_img_multimodal_fusion], axis=1)
    first_layer = tf.nn.dropout(
        tf.contrib.layers.fully_connected(final_concatenation, classification_hidden_size),
        keep_prob=dropout_input
    )
    second_layer = tf.nn.dropout(
        tf.contrib.layers.fully_connected(first_layer, classification_hidden_size),
        keep_prob=dropout_input
    )
    third_layer = tf.nn.dropout(
        tf.contrib.layers.fully_connected(second_layer, classification_hidden_size),
        keep_prob=dropout_input
    )

    return tf.contrib.layers.fully_connected(
        third_layer,
        num_labels,
        activation_fn=None
    )


if __name__ == "__main__":
    random_seed = 12345
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    parser = ArgumentParser()
    parser.add_argument("--train_filename", type=str, required=True)
    parser.add_argument("--dev_filename", type=str, required=True)
    parser.add_argument("--vectors_filename", type=str, required=True)
    parser.add_argument("--img_names_filename", type=str, required=True)
    parser.add_argument("--img_features_filename", type=str, required=True)
    parser.add_argument("--model_save_filename", type=str, required=True)
    parser.add_argument("--max_vocab", type=int, default=300000)
    parser.add_argument("--embeddings_size", type=int, default=300)
    parser.add_argument("--train_embeddings", type=bool, default=True)
    parser.add_argument("--img_features_size", type=int, default=2048)
    parser.add_argument("--rnn_hidden_size", type=int, default=512)
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--multimodal_fusion_hidden_size", type=int, default=512)
    parser.add_argument("--classification_hidden_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--l2_reg", type=float, default=0.000005)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    start_logger(args.model_save_filename + ".train_log")
    atexit.register(stop_logger)

    print("-- Building vocabulary")
    embeddings, token2id, id2token = load_glove(args.vectors_filename, args.max_vocab, args.embeddings_size)
    label2id = {"neutral": 0, "entailment": 1, "contradiction": 2}
    id2label = {v: k for k, v in label2id.items()}
    num_tokens = len(token2id)
    num_labels = len(label2id)
    print("Number of tokens: {}".format(num_tokens))
    print("Number of labels: {}".format(num_labels))

    with open(args.model_save_filename + ".params", mode="w") as out_file:
        json.dump(vars(args), out_file)
        print("Params saved to: {}".format(args.model_save_filename + ".params"))

        with open(args.model_save_filename + ".index", mode="wb") as out_file:
            pickle.dump(
                {
                    "token2id": token2id,
                    "id2token": id2token,
                    "label2id": label2id,
                    "id2label": id2label
                },
                out_file
            )
            print("Index saved to: {}".format(args.model_save_filename + ".index"))

    print("-- Loading training set")
    train_labels, train_premises, train_hypotheses, train_img_names, _, _ = load_vte_dataset(
        args.train_filename,
        token2id,
        label2id
    )

    print("-- Loading development set")
    dev_labels, dev_premises, dev_hypotheses, dev_img_names, _, _ = load_vte_dataset(
        args.dev_filename,
        token2id,
        label2id
    )

    print("-- Loading images")
    image_reader = ImageReader(args.img_names_filename, args.img_features_filename)

    print("-- Building model")
    premise_input = tf.placeholder(tf.int32, (None, None), name="premise_input")
    hypothesis_input = tf.placeholder(tf.int32, (None, None), name="hypothesis_input")
    img_features_input = tf.placeholder(tf.float32, (None, args.img_features_size), name="img_features_input")
    label_input = tf.placeholder(tf.int32, (None,), name="label_input")
    dropout_input = tf.placeholder(tf.float32, name="dropout_input")
    logits = build_simple_vte_model_relu_phi(
        premise_input,
        hypothesis_input,
        img_features_input,
        dropout_input,
        num_tokens,
        num_labels,
        embeddings,
        args.embeddings_size,
        args.train_embeddings,
        args.rnn_hidden_size,
        args.multimodal_fusion_hidden_size,
        args.classification_hidden_size
    )
    loss_function = tf.losses.sparse_softmax_cross_entropy(label_input, logits)
    train_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss_function)
    saver = tf.train.Saver()

    num_examples = train_labels.shape[0]
    num_batches = num_examples // args.batch_size
    dev_best_accuracy = -1
    stopping_step = 0
    best_epoch = None
    should_stop = False

    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1)) as session:
        session.run(tf.global_variables_initializer())

        for epoch in range(args.num_epochs):
            if should_stop:
                break

            print("\n==> Online epoch # {0}".format(epoch + 1))
            progress = Progbar(num_batches)
            batches_indexes = np.arange(num_examples)
            np.random.shuffle(batches_indexes)
            batch_index = 1
            epoch_loss = 0

            for indexes in batch(batches_indexes, args.batch_size):
                batch_premises = train_premises[indexes]
                batch_hypotheses = train_hypotheses[indexes]
                batch_labels = train_labels[indexes]
                batch_img_names = [train_img_names[i] for i in indexes]
                batch_img_features = image_reader.get_features(batch_img_names)

                loss, _ = session.run([loss_function, train_step], feed_dict={
                    premise_input: batch_premises,
                    hypothesis_input: batch_hypotheses,
                    img_features_input: batch_img_features,
                    label_input: batch_labels,
                    dropout_input: args.dropout_ratio
                })
                progress.update(batch_index, [("Loss", loss)])
                epoch_loss += loss
                batch_index += 1
            print("Current mean training loss: {}\n".format(epoch_loss / num_batches))

            print("-- Validating model")
            dev_num_examples = dev_labels.shape[0]
            dev_batches_indexes = np.arange(dev_num_examples)
            dev_num_correct = 0

            for indexes in batch(dev_batches_indexes, args.batch_size):
                dev_batch_premises = dev_premises[indexes]
                dev_batch_hypotheses = dev_hypotheses[indexes]
                dev_batch_labels = dev_labels[indexes]
                dev_batch_img_names = [dev_img_names[i] for i in indexes]
                dev_batch_img_features = image_reader.get_features(dev_batch_img_names)
                predictions = session.run(
                    tf.argmax(logits, axis=1),
                    feed_dict={
                        premise_input: dev_batch_premises,
                        hypothesis_input: dev_batch_hypotheses,
                        img_features_input: dev_batch_img_features,
                        dropout_input: 1.0
                    }
                )
                dev_num_correct += (predictions == dev_batch_labels).sum()
            dev_accuracy = dev_num_correct / dev_num_examples
            print("Current mean validation accuracy: {}".format(dev_accuracy))

            if dev_accuracy > dev_best_accuracy:
                stopping_step = 0
                best_epoch = epoch + 1
                dev_best_accuracy = dev_accuracy
                saver.save(session, args.model_save_filename + ".ckpt")
                print("Best mean validation accuracy: {} (reached at epoch {})".format(dev_best_accuracy, best_epoch))
                print("Best model saved to: {}".format(args.model_save_filename))
            else:
                stopping_step += 1
                print("Current stopping step: {}".format(stopping_step))
            if stopping_step >= args.patience:
                print("Early stopping at epoch {}!".format(epoch + 1))
                print("Best mean validation accuracy: {} (reached at epoch {})".format(dev_best_accuracy, best_epoch))
                should_stop = True
            if epoch + 1 >= args.num_epochs:
                print("Stopping at epoch {}!".format(epoch + 1))
                print("Best mean validation accuracy: {} (reached at epoch {})".format(dev_best_accuracy, best_epoch))
