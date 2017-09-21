import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from Utils import wordFrequency
count = 0
total = 500
k = 50
selected = []

# Load raw data and remove sentences with undesired tokens 
with open("books_in_sentences/books_large_p1.txt", "r") as doc:
    for line in doc:
        line = line.rstrip("\n")
        if "www." in line or "***" in line or "isbn" in line or "# " in line or "copyright" in line or ".co.uk" in line:
            continue
        selected.append(line)
        count += 1
        if count == total:
            break

# Save selected sentences
sentences_save_path = "selected_%d_origin.txt" % total
with open("data/" + sentences_save_path, "w") as doc:
    for line in selected:
        doc.write(line + "\n")

# Calculate the word frequency
vocab = wordFrequency(selected)
words = []
for w, v in vocab:
    if w is not "\n":
        words.append(w)
n_tokens = len(words)
print(n_tokens)
top_k = words[0:k]

# Save top k frequent words 
top_k_saved_path = "top_%d_words.txt" % k
with open("data/" + top_k_saved_path, "w") as doc:
    for w in top_k:
        doc.write(w+"\n")

# Remove top k frequent words
doc_dropped = []
for s in selected:
    print(s)
    sentence_full = s.split(" ")
    sentence_dropped = []
    for w in sentence_full:
        if w not in top_k:
            sentence_dropped.append(w)
    sen = " ".join(sentence_dropped)
    print(sen)
    doc_dropped.append(sen)

# Save the dropped sentences
drop_doc_path = "selected_%d_dropped.txt" % total
with open("data/"+drop_doc_path, "w") as doc:
    for line in doc_dropped:
        doc.write(line + "\n")

# Training LDA
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=None, stop_words='english')
tf = tf_vectorizer.fit_transform(doc_dropped)
lda = LatentDirichletAllocation(n_topics=10, max_iter=5, learning_method='online', learning_offset=50., random_state=0)
lda.fit(tf)
vocab_dict = tf_vectorizer.vocabulary_
components = lda.components_
component_names = tf_vectorizer.get_feature_names()

lda_params = lda.get_params(deep=True)
with open('data/params/lda_params.save', 'wb') as f:
    pickle.dump(lda_params, f, protocol=pickle.HIGHEST_PROTOCOL)

# Save topics and their associate words
for topic_idx, topic in enumerate(components):
    message = "topic_%d" % topic_idx
    print(message)
    idx = topic.argsort()
    with open("data/topics/" + message + ".txt", "w") as doc:
        for i in idx:
            doc.write(component_names[i] + "\n")

tf_sentence_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                         max_features=None,
                                         stop_words='english',
                                         vocabulary=vocab_dict)

# Replace words with their topic id and save the results
topic_sentence_path = "selected_%d_topic_sentence.txt" % total
with open("data/" + topic_sentence_path, "w") as doc:
    for line in selected:
        raw_sentence = line.split(" ")
        sentence = []
        for w in raw_sentence:
            token = w
            if token not in top_k:
                tf = tf_sentence_vectorizer.fit_transform([token])
                topic_distribution = lda.transform(tf)
                t_idx = np.argmax(topic_distribution)
                token = "<topic_%d_>" % t_idx
            sentence.append(token)
        s = " ".join(sentence)
        print(s)
        doc.write(s + "\n")
