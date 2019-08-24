import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

from collections import Counter
import typing as typ
import random
import math


class Word2Vec:
    def __init__(
        self,
        text: str,
        min_word_count=5,
        embedding_size=100,
        num_negative_samples=10,
        context_size=3,
        batch_size=32,
    ):
        self.min_word_count = min_word_count
        self.embedding_size = embedding_size
        self.num_negative_samples = num_negative_samples
        self.context_size = context_size
        self.batch_size = batch_size
        self.load_text(text)

    @staticmethod
    def tokenize_words(text: str) -> typ.List[str]:
        punctuation = set(",.;:?!()")
        text_no_punctuation = "".join(filter(lambda x: x not in punctuation, text))
        words = text_no_punctuation.lower().split()

        return words

    def load_text(self, text: str) -> None:
        words = self.tokenize_words(text)
        word_counter = Counter(words)

        vocab = [
            word for word in word_counter if word_counter[word] >= self.min_word_count
        ]
        self.vocab = set(vocab)
        self.words = list(filter(lambda x: x in self.vocab, words))
        self.vocab_idx_map = {word: i for i, word in enumerate(vocab)}
        self.idx_vocab_map = {i: word for i, word in enumerate(vocab)}
        self.vocab_counts = [word_counter[word] for word in vocab]

    def generate_train_set(self, epochs: int = 10) -> typ.Iterator[typ.Tuple[int, int]]:
        c = self.context_size

        for _ in range(epochs):
            target_indices = list(range(c, len(self.words) - c))
            random.shuffle(target_indices)
            for i in target_indices:
                target_word = self.words[i]
                example_pairs = [
                    (target_word, self.words[i + j]) for j in range(-c, c + 1) if j != 0
                ]
                example_idxs = [
                    (self.vocab_idx_map[target_word], self.vocab_idx_map[context_word])
                    for target_word, context_word in example_pairs
                ]

                for example in example_idxs:
                    yield example

    def forward(self, input_idx, output_idx):
        self.input_embeddings = tf.Variable(
            tf.random_uniform([len(self.vocab), self.embedding_size], -1.0, 1.0),
            name="input_embeddings",
        )
        self.output_embeddings = tf.Variable(
            tf.truncated_normal(
                [len(self.vocab), self.embedding_size],
                stddev=1.0 / math.sqrt(self.embedding_size),
            ),
            name="output_embeddings",
        )

        input_embedding = tf.nn.embedding_lookup(self.input_embeddings, input_idx)
        output_embedding = tf.nn.embedding_lookup(self.output_embeddings, output_idx)

        example_logit = tf.matmul(
            input_embedding, output_embedding, transpose_b=True, name="example_logit"
        )

        negative_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=input_idx,
            num_true=1,
            num_sampled=self.num_negative_samples,
            unique=True,
            range_max=len(self.vocab),
            distortion=0.75,  # raise the unigram distribution to the 3/4 power
            unigrams=self.vocab_counts,
            name="negative_samples",
        )

        negative_samples_tiled = tf.tile(
            tf.expand_dims(negative_samples, dim=0), multiples=[self.batch_size, 1]
        )

        negative_output_embeddings = tf.nn.embedding_lookup(
            self.output_embeddings, negative_samples_tiled
        )
        negative_logits = tf.matmul(
            input_embedding,
            negative_output_embeddings,
            transpose_b=True,
            name="negative_logits",
        )

        return example_logit, negative_logits

    def loss(self, example_logit, negative_logits):
        # We add a small value to prevent log(0)
        example_objective = tf.log(tf.math.sigmoid(example_logit) + 0.0001)
        negative_objective = tf.log(tf.math.sigmoid(-1 * negative_logits) + 0.0001)

        total_objective = (
            tf.reduce_sum(example_objective) + tf.reduce_sum(negative_objective)
        ) / self.batch_size

        # We set the loss to be the negative of the objective function we're trying to maximize
        return -1 * total_objective

    def optimize(self, loss):
        return tf.train.GradientDescentOptimizer(0.001).minimize(loss, name="optimizer")

    def train(self, log_dir: str = "model_logs"):
        graph = tf.Graph()
        with graph.as_default():
            input_idx = tf.placeholder(
                tf.int64, shape=[self.batch_size, 1], name="input_idx"
            )
            output_idx = tf.placeholder(
                tf.int64, shape=[self.batch_size, 1], name="output_idx"
            )
            example_logit, negative_logits = self.forward(input_idx, output_idx)
            loss = self.loss(example_logit, negative_logits)
            optimizer = self.optimize(loss)

            tf.summary.scalar("loss", loss)
            merged = tf.summary.merge_all()

        with tf.Session(graph=graph) as session:
            writer = tf.summary.FileWriter(log_dir, session.graph)
            tf.global_variables_initializer().run()

            input_batch, output_batch = [], []
            batch_count = 0
            for i, example in enumerate(self.generate_train_set()):
                input, output = example
                input_batch.append(input)
                output_batch.append(output)

                if len(input_batch) >= self.batch_size:
                    _, current_loss, summary = session.run(
                        [optimizer, loss, merged],
                        feed_dict={
                            input_idx: np.expand_dims(np.array(input_batch), axis=1),
                            output_idx: np.expand_dims(np.array(output_batch), axis=1),
                        },
                    )
                    input_batch, output_batch = [], []

                    if batch_count % 100 == 0:
                        writer.add_summary(summary, i)
                    if batch_count % 1000 == 0:
                        print(f"{i}:{current_loss}")

            # Setup TensorBoard visualization
            with open(f"{log_dir}/metadata.tsv", "w") as f:
                for i in range(len(self.vocab)):
                    f.write(f"{self.idx_vocab_map[i]}\n")

            saver = tf.train.Saver([self.input_embeddings])
            saver.save(session, f"{log_dir}/model.ckpt")

            config = projector.ProjectorConfig()
            embedding_config = config.embeddings.add()
            embedding_config.tensor_name = self.input_embeddings.name
            embedding_config.metadata_path = f"metadata.tsv"
            projector.visualize_embeddings(writer, config)

            writer.close()
