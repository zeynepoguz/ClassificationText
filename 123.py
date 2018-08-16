import re
import collections
import logging
from nltk.corpus import stopwords
import pandas as pd
import gensim.models.word2vec as w2v
import multiprocessing
import os
import sklearn.manifold
import seaborn as sns

# def tokenizer(text, spliter=" "):
#     newStr = ""
#     if text is not None and len(text)> 0 :
#         newStr = text.split(spliter)
#     return newStr
#
# def clean(text):
#     filter = [',', '?', '"', '-', '\n', '”', '“', ';', '—', '!', '\'', '.', '\t', '\ufeff']
#     for f in filter:
#         if f in text:
#             text = text.replace(f, "").strip()
#
#             print(text)
#     return text
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def commonwords_and_dataframe(stop_words,words):
    wordcount = {}  # dictionary
    for word in words:
        if word not in stop_words:
            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1    # count +1 when it occurs again

    n_print = 100   # number of most common words
    word_counter = collections.Counter(wordcount)

    for word, count in word_counter.most_common(n_print):
        print(word, ": ", count,"times, ", (count/len(words))*100)

    lst = word_counter.most_common(n_print)
    df = pd.DataFrame(lst, columns=['Word', 'Count'])
    df.plot.bar(x='Word', y='Count')


def cleantxt(text, lang):

    if lang is 'english':
        stop_words = set(stopwords.words('english'))  #using nltk.corpus lib
        # print(stop_words)  #print stop words
        words = " ".join(re.findall("[a-zA-Z]+", text)).split() #using rex extract only char
        print(type(words))

    elif lang is 'turkish':
        stop_words = set(stopwords.words('turkish'))    #using nltk.corpus lib
        words = " ".join(re.findall("[abcçdefgğhiıjklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ]+", text)).split()
    else:
        stop_words = set(stopwords.words(lang))     # using nltk.corpus lib- for supported languages >> stopwords.fileids()
        words = " ".join(re.findall("[a-zA-Z]+", text)).split()  # it can be modified depend on the language

    commonwords_and_dataframe(stop_words,words)

    words = [x.lower() for x in words]      # lowercase all strings
    words = [w for w in words if not w in stop_words]       # extract stop words

    return words


def init(filename, language):

    texts = open(filename, encoding='utf-8').read().lower()  # open file
    raw_text = cleantxt(texts, language)    # clean text
    print(type(raw_text))  # <class 'list'>
    print(len(raw_text))   # 104_899
    keyword_set = set(raw_text)
    print(len(keyword_set))  # 10_908

    return keyword_set


_file_dir = ['C:\MyProjects\PythonProjects\ClassifyNews']
_file_name = ['\hp1.txt', '\hp2.txt', '\hp3.txt', '\hp4.txt', '\hp5.txt', '\hp6.txt', '\hp7.txt']
_file_names = []
for f in _file_name:
    for i in _file_dir:
        _file_names.append(i + f)

print(_file_names)

# _file_name1 = "C:\MyProjects\PythonProjects\Keras\hp1.txt"
# _file_name2 = "C:\MyProjects\PythonProjects\Keras\hp2.txt"
_language = 'english'

keywords = []
# keyword = init(_file_name, _language)
for _f_name in _file_names:
    print(_f_name)
    k = init(_f_name, _language)
    k = list(k)
    keywords.append(k)


# print("keyworddddd")
print(keywords)


num_features = 300
min_word_count = 3
num_workers = multiprocessing.cpu_count()
context_size = 7
downsampling = 1e-3

seed = 1

harry2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

harry2vec.build_vocab(keywords)
print("Word2Vec vocabulary length:", len(harry2vec.wv.vocab))

harry2vec.train(keywords, epochs=harry2vec.iter, total_examples=harry2vec.corpus_count)

if not os.path.exists("trained"):
    os.makedirs("trained")

harry2vec.save(os.path.join("trained", "harry2vec.w2v"))
# num_steps=30
#
# batch_size=20
#
# hidden_size=500
#
#
# model = Sequential()
# model.add(Embedding(keywords, hidden_size, input_length=num_steps))
# model.add(LSTM(hidden_size, return_sequences=True))
# model.add(LSTM(hidden_size, return_sequences=True))
# model.add(TimeDistributed(Dense(keywords)))
# model.add(Activation('softmax'))
# model.summary()
#
# model = sklearn.manifold.TSNE(n_components=2, random_state=0)
# np.set_printoptions(suppress=True)
# Y=model.fit_transform(train_data_features)
#
# plt.scatter(Y[:, 0], Y[:, 1], c=clustering_ result, s=290,alpha=.5)
# plt.show()

print("control-1")
tsne = sklearn.manifold.TSNE(n_components=3, random_state=0)
print("control-2")
all_vector_matrix = harry2vec.wv.syn0
print("control-3")
all_vector_matrix_2d = tsne.fit_transform(all_vector_matrix)
print("control-4")

print(harry2vec.most_similar("family"))

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_vector_matrix_2d[harry2vec.wv.vocab[word].index])
            for word in harry2vec.wv.vocab]
    ],
    columns=["word", "x", "y"]
)

print(points.head(10))

sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(20, 12))


def plot_region(x_bounds, y_bounds):

    slice = points[
        (x_bounds[0] <= points.x) & (points.x <= x_bounds[1]) &
        (y_bounds[0] <= points.y) & (points.y <= y_bounds[1])
        ]
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)


print(points.x, points.y)
plot_region(x_bounds=(10.0, 26.2), y_bounds=(-24.0, 26.4))
