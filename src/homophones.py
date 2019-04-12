import json
import os
import pathlib

import tensorflow as tf

import encoder
import model
import matplotlib.pyplot as plt


class Homophones:
    def __init__(self, model_name='117M'):
        self._session = tf.Session(graph=tf.Graph())
        self._enc = encoder.get_encoder(model_name)
        hparams = model.default_hparams()
        with open(os.path.join('models', model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        with self._session.graph.as_default():
            self._context_ph = tf.placeholder(tf.int32, [None])
            lm_output = model.model(hparams=hparams, X=self._context_ph[None, :], past=None, reuse=tf.AUTO_REUSE)
            self._logits = lm_output['logits']
            self._probs = tf.nn.softmax(self._logits, axis=-1)
            saver = tf.train.Saver()

            ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
            saver.restore(self._session, ckpt)

        self._hyperparams = hparams

    def predict_probabilities(self, tokens):
        return self._session.run(self._probs, feed_dict={self._context_ph: tokens})[0]

    def extract_context(self, tokens, position):
        start = position - 8 #self._hyperparams.n_ctx
        if start < 0:
            start = 0
        return tokens[start:position]

    def check_confusion(self, text, word_a, word_b):
        encoded_text = self._enc.encode(text)
        word_a_id = self._enc.encode(" " + word_a)[0]
        word_b_id = self._enc.encode(" " + word_b)[0]
        homophones = [word_a_id, word_b_id]
        print(encoded_text)
        print(homophones)

        all_probs = []

        occurences = sum(map(lambda x: 1 if x in homophones else 0, encoded_text))
        print("# occurences", occurences)

        for index, id in enumerate(encoded_text):
            if id in homophones:
                window = self.extract_context(encoded_text, index)
                probs = self.predict_probabilities(window)[-1]

                probs_a = probs[word_a_id]
                probs_b = probs[word_b_id]
                total = probs_a + probs_b
                probs_a = probs_a / total
                probs_b = probs_b / total

                if id == word_a_id:
                    prob_true = probs_a
                else:
                    prob_true = probs_b

                all_probs.append(prob_true)

        return all_probs


program = Homophones("117M")


def make_histogram(word_a, word_b):
    text = pathlib.Path("sample.txt").read_text()
    conf = program.check_confusion(text, word_a, word_b)

    plt.hist(conf)
    plt.title("%s / %s" % (word_a, word_b))
    plt.xlabel("probability of correct guess")
    plt.ylabel("# occurence")
    plt.show()


make_histogram("to", "two")
#make_histogram("is", "was")
#make_histogram("from", "to")
#make_histogram("person", "man")
