from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def wordFrequency(path):
    vocab = {}
    with open(path, "r") as doc:
        for line in doc:
            sentence = line.split(" ")
            for word in sentence:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 0
    return sorted(vocab.items(), key=lambda d: d[1], reverse=True)


def ldaVocab(path, num_topics, num_features=None):
    print("extract topic")
    content = []
    with open(path, "r") as doc:
        for line in doc:
            content.append(line)
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=num_features,
                                    stop_words='english')
    tf = tf_vectorizer.fit_transform(content)
    lda = LatentDirichletAllocation(n_topics=num_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    return lda.components_, tf_vectorizer.get_feature_names()

