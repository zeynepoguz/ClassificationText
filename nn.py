import textacy
import keras
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
import textacy.keyterms
from keras.preprocessing.text import text_to_word_sequence
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
from keras import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split


print("-------------------------------------------------------\n")

texts = open("C:\MyProjects\PythonProjects\Keras\sherlock.txt",encoding='utf-8').read().lower()
print(texts)

words = set(text_to_word_sequence(texts))
vocab_size = len(words)

txt = keras.preprocessing.text.one_hot(texts, vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                                     lower=True, split=' ')
print("-------\n")
print(txt)

data = array(txt)
print(data)
# one hot encode
encoded = to_categorical(data)
print(encoded)
# invert encoding
inverted = argmax(encoded[0])
print("****************************************")
print(inverted)


# embedding_layer = Embedding(1000, 64)
#
# # Number of words to consider as features
# max_features = 10000
# # Cut texts after this number of words
# # (among top max_features most common words)
# maxlen = 20
#
# train_data, test_data = train_test_split(a, test_size=0.2)
# print(len(train_data))
# print(len(test_data))
# print(train_data[10])
#
# model = Sequential()
# model.add(Dense(32, input_dim=784))
# model.add(Activation('relu'))
#
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
# model.summary()






