#! /usr/bin/env python
from time import time
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
import pandas as pd

import math
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
import statistics

import tensorflow as tf

from ugcn.model import *
from util import *

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", default="KAGGLE", help="Output directory name")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--num_epochs", default=50, type=int, help="Number of training epochs")
    parser.add_argument("--num_sampled", default=256, type=int, help="Sampled softmax length to embedding")
    parser.add_argument("--hidden_size", default=128, type=int, help="The hidden layer size")
    parser.add_argument("--num_conv_layers", default=2, type=int, help="Number of stacked graph convolution layers")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
    parser.add_argument("--no_soft_placement", action='store_true', help="Disallow device soft device placement")
    parser.add_argument("--log_device_placement", action='store_true', help="Log placement of ops on devices")
    parser.add_argument("--load_epoch", default=0, type=int, help="Load previous state if set")
    parser.add_argument("--test_only", action='store_true', help="Print test result and exit")
    args = parser.parse_args()
    print(args)

    print("Loading data...")
    graphs: "list[S2VGraph]"
    label_map: "dict[int, int]"
    graph_pos: "dict[int, int]"
    graphs, label_map, graph_pos = load_cached_data()
    print("Loading data finished.")

    graph_labels = np.array([graph.label for graph in graphs])
    graph_features = np.array([graph.graph_features for graph in graphs])
    feature_dim_size = graphs[0].node_features.shape[1] + graphs[0].node_centrality.shape[1]
    num_classes = len(label_map)
    print(f"node features dimension: {feature_dim_size}")

    def get_graphpool(batch_graph):
        start_idx = [0]
        # compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            elem.extend([1] * len(graph.g))
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])

        elem = np.array(elem)
        idx = np.array(idx)

        graph_pool = coo_matrix((elem, (idx[:, 0], idx[:, 1])), shape=(len(batch_graph), start_idx[-1]))
        return graph_pool

    graph_pool = get_graphpool(graphs)
    vocab_size = graph_pool.shape[1]

    def get_batch_data(selected_idx):
        batch_graph = [graphs[idx] for idx in selected_idx]
        c_concat = np.concatenate([graph.node_centrality for graph in batch_graph], 0)
        X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
        X_concat = np.concatenate((X_concat, c_concat), axis=1)

        X_concat = coo_matrix(X_concat)
        X_concat = sparse_to_tuple(X_concat)

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])

        Adj_block_idx = np.concatenate(edge_mat_list, 1)
        Adj_block_elem = np.ones(Adj_block_idx.shape[1])

        # self-loop
        num_node = start_idx[-1]
        self_loop_edge = np.array([range(num_node), range(num_node)])
        elem = np.ones(num_node)
        Adj_block_idx = np.concatenate([Adj_block_idx, self_loop_edge], 1)
        Adj_block_elem = np.concatenate([Adj_block_elem, elem], 0)

        Adj_block = coo_matrix((Adj_block_elem, Adj_block_idx), shape=(num_node, num_node))
        Adj_block = sparse_to_tuple(Adj_block)

        input_y = [np.where(graph_pool.getrow(i).toarray()[0] == 1)[0] for i in selected_idx]
        input_y = np.reshape(np.concatenate(input_y), (-1, 1))

        return X_concat, Adj_block, input_y

    def get_graph_embeddings(node_embeddings):
        graph_embeddings = graph_pool.dot(node_embeddings)
        graph_embeddings = np.concatenate((graph_embeddings, graph_features), 1)
        return graph_embeddings

    train_all_idx, _ = train_idx(graphs, labeled_only=False)
    train_intr_idx, _ = train_idx(graphs, labeled_only=True)
    test_intr_idx, test_real_idx = test_idx(graph_pos)

    with tf.Graph().as_default():
        tf_sess_conf = tf.compat.v1.ConfigProto(allow_soft_placement=(not args.no_soft_placement),
                                                log_device_placement=args.log_device_placement)
        tf_sess_conf.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=tf_sess_conf) as tf_sess:
            tf_sess.as_default()

            global_step = tf.Variable(0, name="global_step", trainable=False)

            model = UnsupGCN(feature_dim_size=feature_dim_size, hidden_size=args.hidden_size,
                             num_conv_layers=args.num_conv_layers,
                             vocab_size=vocab_size, num_sampled=args.num_sampled)

            # Define Training procedure
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate)
            gradients = optimizer.compute_gradients(model.total_loss)
            optimizer = optimizer.apply_gradients(gradients, global_step=global_step)

            # Initialize all variables
            tf_sess.run(tf.compat.v1.global_variables_initializer())
            tf_graph = tf.compat.v1.get_default_graph()

            def train():
                total_loss = 0.
                for selected_idx in tqdm(batch(shuffle(train_all_idx), batch_size=args.batch_size), desc="train"):
                    X_concat, Adj_block, input_y = get_batch_data(selected_idx)
                    feed_dict = {
                        model.Adj_block: Adj_block,
                        model.X_concat: X_concat,
                        model.num_features_nonzero: X_concat[1].shape,
                        model.dropout: args.dropout,
                        model.input_y: input_y
                    }
                    _, step, loss = tf_sess.run([optimizer, global_step, model.total_loss], feed_dict)
                    total_loss += loss
                return total_loss

            def evaluate():
                node_embeddings = tf_sess.run(tf_graph.get_tensor_by_name("embedding/node_embeddings:0"))
                graph_embeddings = get_graph_embeddings(node_embeddings)

                accuracy = []
                for train_idx, test_idx in kfold_train_idx(graphs, n_splits=10, labeled_only=True):
                    train_embed = graph_embeddings[train_idx]
                    test_embed = graph_embeddings[test_idx]
                    train_labels = graph_labels[train_idx].astype(int)
                    test_labels = graph_labels[test_idx].astype(int)

                    cls = LogisticRegression(solver="liblinear", tol=0.001, multi_class="auto")
                    cls.fit(train_embed, train_labels)
                    acc = cls.score(test_embed, test_labels)
                    accuracy.append(acc)

                print(f"Evaluate:   {'   '.join([f'{acc*100:5.2f}' for acc in accuracy])}")
                return statistics.mean(accuracy), statistics.stdev(accuracy)

            def test():
                node_embeddings = tf_sess.run(tf_graph.get_tensor_by_name("embedding/node_embeddings:0"))
                graph_embeddings = get_graph_embeddings(node_embeddings)

                train_embed = graph_embeddings[train_intr_idx]
                train_labels = graph_labels[train_intr_idx].astype(int)

                cls = LogisticRegression(solver="liblinear", tol=0.001, multi_class="auto")
                cls.fit(train_embed, train_labels)

                test_embed = graph_embeddings[test_intr_idx]

                test_labels_est = cls.predict(test_embed)
                test_labels_est = [(label_map[l] if l in label_map else l) for l in test_labels_est]

                test_acc = test_accuracy(test_real_idx, test_labels_est)
                return test_acc, test_labels_est

            """main process"""
            out_dir: Path = project_root/"runs/uGCN"/args.model_name
            out_dir.mkdir(exist_ok=True, parents=True)
            print(f"Using {out_dir}")

            global_args_key = ["num_sampled", "hidden_size", "num_conv_layers", "dropout"]
            epoch_args_key = ["learning_rate", "batch_size"]
            checkpoints_key = ["time", "train_acc", "train_acc_stdev", "train_loss", "test_acc"]

            def filter_args(arg: dict, filter: list):
                return {k: v for k, v in arg.items() if k in filter}

            if (out_dir/"args.json").is_file():
                with open(out_dir/"args.json", "r") as f:
                    cargs = filter_args(vars(args), global_args_key)
                    sargs = filter_args(json.load(f), global_args_key)
                    if cargs != sargs:
                        print('args mismatch!')
                        print('current args:', cargs)
                        print('saved args:', sargs)
                        raise ValueError(f"runs/uGCN/{args.model_name}/args.json is not same with current args, delete it to dismiss.")
            else:
                with open(out_dir/"args.json", "w") as f:
                    json.dump(filter_args(vars(args), global_args_key), f, indent=4)

            checkpoints = pd.DataFrame(columns=[*epoch_args_key, *checkpoints_key])
            checkpoints.index.name = "epoch"

            tf_saver = tf.compat.v1.train.Saver()
            if args.load_epoch > 0:
                epoch_dir = out_dir/f"{args.load_epoch}"
                tf_saver.restore(tf_sess, str(epoch_dir/"model"))
                checkpoints = pd.concat([checkpoints, pd.read_csv(epoch_dir/"checkpoints.csv", index_col="epoch")])
                test_acc, label_est = test()
                print(f"state loaded from epoch {args.load_epoch} - test_acc: {test_acc*100:.2f}%")
                if args.test_only:
                    pd.DataFrame(data={"Id": test_real_idx, "Category": label_est}).to_csv(epoch_dir/"test_sample.csv", index=False)
            elif args.test_only:
                raise RuntimeError("args.test_only must be set with args.load_epoch")

            if not args.test_only:
                for epoch in range(args.load_epoch + 1, args.num_epochs + 1):
                    start_time = time()
                    train_loss = train()
                    train_acc, train_acc_stdev = evaluate()
                    time_spend = time() - start_time

                    test_acc, label_est = test()

                    print(f"| epoch {epoch:3d} | time: {time_spend:7.2f}s | train_acc: {train_acc*100:5.2f}% [stdev={train_acc_stdev*100:5.2f}] " +
                          f"| train_loss: {train_loss:11.6f} | test_acc: {test_acc*100:5.2f}%")
                    checkpoints.loc[epoch] = {**{k: v for k, v in zip(checkpoints_key, [time_spend, train_acc, train_acc_stdev, train_loss, test_acc])},
                                              **filter_args(vars(args), epoch_args_key)}

                    epoch_dir = out_dir/f"{epoch}"
                    epoch_dir.mkdir(exist_ok=True)
                    tf_saver.save(tf_sess, str(epoch_dir/"model"))
                    checkpoints.to_csv(epoch_dir/"checkpoints.csv")
                    pd.DataFrame(data={"Id": test_real_idx, "Category": label_est}).to_csv(epoch_dir/"test_sample.csv", index=False)
