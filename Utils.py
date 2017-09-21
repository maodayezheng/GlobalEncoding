from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def wordFrequency(doc):
    vocab = {}
    for line in doc:
            sentence = line.split(" ")
            for word in sentence:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 0
    return sorted(vocab.items(), key=lambda d: d[1], reverse=True)


def ldaTopicToWords(sequence, lda, topics):
    topic_sentence = sequence
    sentence = []
    components = lda.components_
    if isinstance(topic_sentence, str):
        topic_sentence = topic_sentence.split(" ")
    for w in topic_sentence:
        token = w
        if "<topic_" in token:
            topic_id = token.split("_")
            topic_id = int(topic_id[1])
            words_dist = np.array(components[topic_id])
            words_dist = words_dist / np.sum(words_dist)
            word_list = topics[topic_id]
            word_id = np.random.choice(len(words_dist), p=words_dist)
            word_id = int(word_id)
            token = word_list[word_id]
        sentence.append(token)
    return " ".join(sentence)
