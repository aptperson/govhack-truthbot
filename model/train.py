import pickle

import numpy as np
import pandas as pd

import tensorflow as tf
import tflearn

num_docs = 10000

doc_embedding_size = 20
word_embedding_size = 50

def load_data():
	with open('tweet_vocab.pkl', 'rb') as f:
		vocab = pickle.load(f)

	tweet_df = pd.read_pickle('tweet_data_tokenized.pkl')
	tweet_df = tweet_df.sort_values(by='id', ascending=False)
	tweet_df = tweet_df.head(num_docs)
	tweet_df = tweet_df.reset_index(drop=True)

	return tweet_df, vocab


def build_context(v, context_size=3):
    k = context_size
    v = np.lib.pad(v, (k,0), 'constant', constant_values=(0,))
    for i in range(len(v) - k):
        yield v[i:i+k],v[i+k]


def prepare_contexts(tweet_df):
	data = [(tweet_id, context, target) for tweet_id, words in tweet_df.tokenized.iteritems() for context, target in build_context(words)]

	docs     = np.array([x[0] for x in data])
	contexts = np.array([x[1] for x in data])
	targets  = np.array([x[2] for x in data])

	return docs, contexts, targets


def build_model(num_words, num_docs):
    input_layer1 = tflearn.input_data(shape=[None, 1])
    input_layer2 = tflearn.input_data(shape=[None, 3])

    embedding_layer1, = tflearn.embedding(input_layer1, input_dim=num_docs, output_dim=doc_embedding_size)
    e1, e2, e3 = tflearn.embedding(input_layer2, input_dim=num_words, output_dim=word_embedding_size)

    embedding_layer = tflearn.merge([embedding_layer1, e1, e2, e3], mode='concat')
    
    softmax = tflearn.fully_connected(embedding_layer, num_words, activation='softmax')

    optimizer = tflearn.optimizers.Adam(learning_rate=0.001)
    metric = tflearn.metrics.Accuracy()
    net = tflearn.regression(softmax, optimizer=optimizer, metric=metric, batch_size=32,
                             loss='categorical_crossentropy')

    model = tflearn.DNN(net, tensorboard_verbose=0)
    return model


def train_model(model, docs, contexts, targets, num_words):
    X1 = docs[:,np.newaxis]
    X2 = contexts

    Y = tflearn.data_utils.to_categorical(targets, num_words)
    model.fit([X1, X2], Y, n_epoch=20, show_metric=True, run_id="embedding_model")


if __name__ == '__main__':
	tweet_df, vocab = load_data()
	embedding_index_map = tweet_df[['id', 'text']]
	embedding_index_map.to_pickle('embedding_index_map.pkl')
	docs, contexts, targets = prepare_contexts(tweet_df)
	model = build_model(num_words = len(vocab), num_docs = len(tweet_df))
	train_model(model, docs, contexts, targets, num_words = len(vocab))
	model.save('embedding_model.cpkt')
