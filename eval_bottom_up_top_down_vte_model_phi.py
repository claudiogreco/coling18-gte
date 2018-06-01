import atexit
import csv
import json
import os
import pickle
import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from datasets import ImageReader, load_vte_dataset
from train_bottom_up_top_down_vte_model_phi import build_bottom_up_top_down_vte_model_phi
from utils import batch
from utils import start_logger, stop_logger

if __name__ == "__main__":
    random_seed = 12345
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    parser = ArgumentParser()
    parser.add_argument("--test_filename", type=str, required=True)
    parser.add_argument("--model_filename", type=str, required=True)
    parser.add_argument("--img_names_filename", type=str, required=True)
    parser.add_argument("--img_features_filename", type=str, required=True)
    parser.add_argument("--result_filename", type=str, required=True)
    args = parser.parse_args()
    start_logger(args.result_filename + ".log")
    atexit.register(stop_logger)

    print("-- Loading params")
    with open(args.model_filename + ".params", mode="r") as in_file:
        params = json.load(in_file)

    print("-- Loading index")
    with open(args.model_filename + ".index", mode="rb") as in_file:
        index = pickle.load(in_file)
        token2id = index["token2id"]
        id2token = index["id2token"]
        label2id = index["label2id"]
        id2label = index["id2label"]
        num_tokens = len(token2id)
        num_labels = len(label2id)

    print("-- Loading test set")
    test_labels, test_padded_premises, test_padded_hypotheses, test_img_names, test_original_premises, test_original_hypotheses = \
        load_vte_dataset(
            args.test_filename,
            token2id,
            label2id
        )

    print("-- Loading images")
    image_reader = ImageReader(args.img_names_filename, args.img_features_filename)

    print("-- Restoring model")
    premise_input = tf.placeholder(tf.int32, (None, None), name="premise_input")
    hypothesis_input = tf.placeholder(tf.int32, (None, None), name="hypothesis_input")
    img_features_input = tf.placeholder(tf.float32, (None, params["num_img_features"], params["img_features_size"]),
                                        name="img_features_input")
    label_input = tf.placeholder(tf.int32, (None,), name="label_input")
    dropout_input = tf.placeholder(tf.float32, name="dropout_input")
    logits = build_bottom_up_top_down_vte_model_phi(
        premise_input,
        hypothesis_input,
        img_features_input,
        dropout_input,
        num_tokens,
        num_labels,
        None,
        params["embeddings_size"],
        params["num_img_features"],
        params["img_features_size"],
        params["train_embeddings"],
        params["rnn_hidden_size"],
        params["classification_hidden_size"],
        params["multimodal_fusion_hidden_size"]
    )
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1)) as session:
        saver.restore(session, args.model_filename + ".ckpt")

        print("-- Evaluating model")
        test_num_examples = test_labels.shape[0]
        test_batches_indexes = np.arange(test_num_examples)
        test_num_correct = 0
        y_true = []
        y_pred = []

        with open(args.result_filename + ".predictions", mode="w") as out_file:
            writer = csv.writer(out_file, delimiter="\t")
            for indexes in batch(test_batches_indexes, params["batch_size"]):
                test_batch_premises = test_padded_premises[indexes]
                test_batch_hypotheses = test_padded_hypotheses[indexes]
                test_batch_labels = test_labels[indexes]
                batch_img_names = [test_img_names[i] for i in indexes]
                batch_img_features = image_reader.get_features(batch_img_names)
                test_original_premises = np.array(test_original_premises)
                test_original_hypotheses = np.array(test_original_hypotheses)
                test_batch_original_premises = test_original_premises[indexes]
                test_batch_original_hypotheses = test_original_hypotheses[indexes]
                predictions = session.run(
                    tf.argmax(logits, axis=1),
                    feed_dict={
                        premise_input: test_batch_premises,
                        hypothesis_input: test_batch_hypotheses,
                        img_features_input: batch_img_features,
                        dropout_input: 1.0
                    }
                )
                test_num_correct += (predictions == test_batch_labels).sum()
                for i in range(len(indexes)):
                    writer.writerow(
                        [
                            id2label[test_batch_labels[i]],
                            id2label[predictions[i]],
                            " ".join([id2token[id] for id in test_batch_premises[i] if id != token2id["#pad#"]]),
                            " ".join([id2token[id] for id in test_batch_hypotheses[i] if id != token2id["#pad#"]]),
                            batch_img_names[i],
                            test_batch_original_premises[i],
                            test_batch_original_hypotheses[i]
                        ]
                    )
                    y_true.append(id2label[test_batch_labels[i]])
                    y_pred.append(id2label[predictions[i]])
        test_accuracy = test_num_correct / test_num_examples
        print("Mean test accuracy: {}".format(test_accuracy))
        y_true = pd.Series(y_true, name="Actual")
        y_pred = pd.Series(y_pred, name="Predicted")
        confusion_matrix = pd.crosstab(y_true, y_pred, margins=True)
        confusion_matrix.to_csv(args.result_filename + ".confusion_matrix")

        data = pd.read_csv(
            args.result_filename + ".predictions",
            sep="\t",
            header=None,
            names=["gold_label", "prediction", "premise_toks", "hypothesis_toks", "jpg", "premise", "hypothesis"]
        )

        print("Overall accuracy: {}".format(accuracy_score(data["gold_label"], data["prediction"])))

        data_entailment = data.loc[data["gold_label"] == "entailment"]
        print("Accuracy for 'entailment': {}".format(
            accuracy_score(data_entailment["gold_label"], data_entailment["prediction"])))

        data_contradiction = data.loc[data["gold_label"] == "contradiction"]
        print("Accuracy for 'contradiction': {}".format(
            accuracy_score(data_contradiction["gold_label"], data_contradiction["prediction"])))

        data_neutral = data.loc[data["gold_label"] == "neutral"]
        print(
            "Accuracy for 'neutral': {}".format(accuracy_score(data_neutral["gold_label"], data_neutral["prediction"])))
