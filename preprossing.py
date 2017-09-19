from Utils import wordFrequency, ldaVocab
total = 0
selected = []

with open("books_in_sentences/books_large_p1.txt", "r") as doc:
    for line in doc:
        if "www." in line or "***" in line or "isbn" in line or "# " in line or "copyright" in line or ".co.uk" in line:
            continue
        selected.append(line)
        total += 1
        if total == 500:
            break

with open("data/selected_500_raw.txt", "w") as doc:
    for line in selected:
        doc.write(line + "\n")

vocab = wordFrequency("data/selected_500_raw.txt")
words = []
for k, v in vocab:
    if k is not "\n":
        words.append(k)
n_tokens = len(words)
print(n_tokens)
k = int(0.1*n_tokens)
top_k = words[0:50]
with open("data/selected_top_50_vocab.txt", "w") as doc:
    for w in top_k:
        doc.write(w+"\n")

with open("data/selected_500_raw.txt", "r") as raw, open("data/selected_500.txt", "w") as doc:
    for line in raw:
        raw_sentence = line.split(" ")
        sentence = []
        for w in raw_sentence:
            if w not in top_k:
                sentence.append(w)
        if len(sentence) != 0:
            s = " ".join(sentence)
            doc.write(s + "\n")

components, component_names = ldaVocab("data/selected_500.txt", 10)

for topic_idx, topic in enumerate(components):
    message = "topic_%d" % topic_idx
    print(message)
    idx = topic.argsort()
    words = []
    with open("data/selected_500_" + message + ".txt", "w") as doc:
        for i in idx:
            doc.write(component_names[i] + "\n")

print()
