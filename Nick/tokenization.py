import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Suess import stories

vocab_size = 20000
training_size = 10000
max_length = 100
trunc_type = 'post'
padding_type = 'post'
embedding_dim = 16

sentences = []

for story in stories:
    for sentence in story.split('\n'):
        sentences.append(sentence)


training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

tokenizer = Tokenizer(num_words = vocab_size, oov_token = "<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

import numpy as np
training_padded = np.array(training_padded)
testing_padded = np.array(testing_padded)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# model.summary()

# num_epochs = 30
# history = model.fit(training_padded, epochs=num_epochs, validation_data=(testing_padded), verbose=2)