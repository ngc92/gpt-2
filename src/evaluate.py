import json
import os
import numpy as np

import tensorflow as tf

import encoder
import model


def evaluate_sequence(raw_texts, model_name='117M'):
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))


    with tf.Session(graph=tf.Graph()) as sess:
        context_ph = tf.placeholder(tf.int32, [1, None])

        lm_output = model.model(hparams=hparams, X=context_ph, past=None, reuse=tf.AUTO_REUSE)
        logits = lm_output['logits']
        probs = tf.nn.softmax(logits, axis=-1)
        # P_i = exp(L_i) / sum(exp(L_i))
        # log P_i = L_1 - log(sum(exp(L_i))

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        perplexities = []
        guess_alt = []
        guess_ref = []

        reference_text = enc.encode(raw_texts[0])

        # first, do the reference text
        reference_probs = sess.run(probs, feed_dict={context_ph: [reference_text]})[0]
        true_probs = np.asarray([reference_probs[i, c] for i, c in enumerate(reference_text[1:])])

        reference_perplexity = np.exp(-np.mean(np.log(true_probs)))
        perplexities.append(reference_perplexity)

        encoded_texts = [enc.encode(raw_text) for raw_text in raw_texts[1:]]

        for encoded_text in encoded_texts:
            next_probs = sess.run(probs, feed_dict={context_ph: [encoded_text]})[0]
            true_probs = np.asarray([next_probs[i, c] for i, c in enumerate(encoded_text[1:])])

            perplexity = np.exp(-np.mean(np.log(true_probs)))
            perplexities.append(perplexity)

            correct_guess_alt = []
            correct_guess_ref = []

            for i in range(len(true_probs)):
                if encoded_text[i+1] != reference_text[i+1]:
                    # true_probs_ref gives us the probability of having guessed the correct word
                    original_word = enc.decode([reference_text[i+1]])[1:]
                    changed_word = enc.decode([encoded_text[i+1]])[1:]

                    alt_true = next_probs[i, reference_text[i + 1]]
                    alt_conf = next_probs[i, encoded_text[i + 1]]

                    print("-------")
                    print(" " + changed_word + " vs " + original_word)
                    print("absolute", alt_conf, " ", alt_true)
                    tp = alt_true + alt_conf
                    print("relative", alt_conf / tp, " ", alt_true / tp)

                    correct_guess_alt += [alt_true / tp]

                    # now the reverse. look at the reference probability
                    ref_true = reference_probs[i, reference_text[i+1]]
                    ref_conf = reference_probs[i, encoded_text[i+1]]
                    tp = ref_true + ref_conf
                    print("original", ref_conf / tp, " ", ref_true / tp)

                    correct_guess_ref += [ref_true / tp]

            guess_alt.append(np.exp(-np.mean(np.log(correct_guess_alt))))
            guess_ref.append(np.exp(-np.mean(np.log(correct_guess_ref))))

    return perplexities, guess_alt, guess_ref


TEST_TEXT = "Three Arch Rocks National Wildlife Refuge is a U.S. National Wildlife Refuge off the northern Oregon " \
            "Coast. It is located on the central coast of Tillamook County, in the northwestern part of Oregon. It is " \
            "one of six National Wildlife Refuges within the Oregon Coast National Wildlife Refuge Complex and was " \
            "the first National Wildlife Refuge west of the Mississippi River. In 1970 the Refuge was designated as " \
            "wilderness. It is one of the smallest wilderness areas in the United States.[3] "\
            "Three Arch Rocks consists of 15 acres (6 ha) on three large and six small rocky islands located about a " \
            "half mile (1 km) offshore from Oceanside. It is one of the smallest designated wilderness areas in the " \
            "U.S., but features the largest colony of breeding tufted puffins and the largest common murre colony " \
            "south of Alaska. It is the only northern Oregon pupping site for the threatened Steller sea lion.[4]" \
            "The refuge was established by President Theodore Roosevelt after being persuaded by two young " \
            "conservationists — William L. Finley and Herman Bohlman — who studied and photographed Three Arch Rocks " \
            "from Oceanside beginning in 1901. They recorded hunters killing dozens of sea lions at a time for skin " \
            "and oil, and sportsmen shooting seabirds purely for sport. Due to a scarcity of regional chicken farms " \
            "at the time, seabird eggs were priced at up to a dollar per dozen, encouraging egg harvesting and " \
            "reducing the bird colony population. Finley and Bohlman suggested a wildlife refuge to Roosevelt to " \
            "protect dwindling populations and ensure survival of seabird and marine mammal populations. Roosevelt " \
            "declared the Three Rocks area a National Wildlife Refuge in 1907.[4] In 1970 the United States Congress " \
            "designated the Refuge wilderness. In 1994, there was a sighting of a group of 2 or 3 North Pacific right " \
            "whales, the most rare and endangered of all large whales at the Rocks.[5] " \
            "The Three Arch Rocks Refuge has provided protection for Oregon's largest seabird nesting colony of more " \
            "than 230,000 birds since October 14, 1907.[4] The entire Oregon Coast National Wildlife Refuge Complex " \
            "protect over a million nesting seabirds, including common murres, tufted puffins, cormorants, " \
            "and storm-petrels. "


perplexity, two_word, reference = evaluate_sequence([TEST_TEXT,
                                                     TEST_TEXT.replace("to ", "two "),
                                                     TEST_TEXT.replace("is ", "was ")])
print(perplexity)
print(two_word)
print(reference)

