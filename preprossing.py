import numpy as np
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
        if "www." in line or "***" in line or "isbn" in line or "# " in line or "copyright" in line or ".co.uk" in line:
            continue
        selected.append(line)
        count += 1
        if count == total:
            break

# Save selected sentences
sentences_save_path = "selected" + str(total) + "_origin.txt"
with open("data/"+ sentences_save_path, "w") as doc:
    for line in selected:
        doc.write(line)

vocab = wordFrequency("data/" + sentences_save_path)
words = []
for k, v in vocab:
    if k is not "\n":
        words.append(k)
n_tokens = len(words)
print(n_tokens)
k = int(0.1*n_tokens)
top_k = words[0:50]

# Save top k frequent words 
top_k_saved_path = "top_" + str(k) + "words.txt"
with open("data/" + top_k_saved_path, "w") as doc:
    for w in top_k:
        doc.write(w+"\n")

# Remove top k frequent words
with open("data"+sentences_save_path, "r") as raw, open("data/selected_500.txt", "w") as doc:
    for line in raw:
        raw_sentence = line.split(" ")
        sentence = []
        for w in raw_sentence:
            if w not in top_k:
                sentence.append(w)
        if len(sentence) != 0:
            s = " ".join(sentence)
            doc.write(s + "\n")

# Training LDA
content = []
tf_vectorizer = None
lda = None
with open("data/selected_500.txt", "r") as doc:
    for line in doc:
        content.append(line)
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=None,
                                    stop_words='english')
    tf = tf_vectorizer.fit_transform(content)
    lda = LatentDirichletAllocation(n_topics=100, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
vocab_dict = tf_vectorizer.vocabulary_
components = lda.components_
component_names = tf_vectorizer.get_feature_names()

# Save topics and their associate words 
for topic_idx, topic in enumerate(components):
    message = "topic_%d" % topic_idx
    print(message)
    idx = topic.argsort()
    words = []
    with open("data/selected_500_" + message + ".txt", "w") as doc:
        for i in idx:
            doc.write(component_names[i] + "\n")

tf_sentence_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                         max_features=None,
                                         stop_words='english',
                                         vocabulary=vocab_dict)

# Replace words with their topic id and save the results
with open("data/selected_500_raw.txt", "r") as raw, open("data/selected_500.txt", "w") as doc:
    for line in raw:
        raw_sentence = line.split(" ")
        sentence = []
        for w in raw_sentence:
            token = w
            if token not in top_k:
                tf = tf_sentence_vectorizer.fit_transform([token])
                topic_distribution = lda.transform(tf)
                t_idx = np.argmax(topic_distribution)
                token = "<topic_" + str(t_idx) + ">"
            sentence.append(token)
        s = " ".join(sentence)
        print(s)
        doc.write(s + "\n")
