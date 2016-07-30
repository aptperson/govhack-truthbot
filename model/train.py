import pickle

import numpy as np
import pandas as pd

import tensorflow as tf
import tflearn

num_docs = 10000

doc_embedding_size = 100
word_embedding_size = 100

def load_data(num_docs=num_docs):
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

	np.random.shuffle(data)

	docs     = np.array([x[0] for x in data])
	contexts = np.array([x[1] for x in data])
	targets  = np.array([x[2] for x in data])

	return docs, contexts, targets


def build_model(num_words, num_docs,
		doc_embedding_size=doc_embedding_size, word_embedding_size=word_embedding_size):
    input_layer1 = tflearn.input_data(shape=[None, 1])
    input_layer2 = tflearn.input_data(shape=[None, 3])

    d1, = tflearn.embedding(input_layer1, input_dim=num_docs, output_dim=doc_embedding_size)
    w1, w2, w3 = tflearn.embedding(input_layer2, input_dim=num_words, output_dim=word_embedding_size)

    embedding_layer = tflearn.merge([d1, w1, w2, w3], mode='concat')
    softmax = tflearn.fully_connected(embedding_layer, num_words, activation='softmax')

    optimizer = tflearn.optimizers.Adam(learning_rate=0.001)
    # optimizer = tflearn.optimizers.SGD(learning_rate=0.1)

    metric = tflearn.metrics.Accuracy()
    net = tflearn.regression(softmax, optimizer=optimizer, metric=metric, batch_size=16,
                             loss='categorical_crossentropy')

    model = tflearn.DNN(net, tensorboard_verbose=0, checkpoint_path='embedding_model')
    return model


def train_model(model, docs, contexts, targets, num_words):
    X1 = docs[:,np.newaxis]
    X2 = contexts

    Y = tflearn.data_utils.to_categorical(targets, num_words)
    model.fit([X1, X2], Y, n_epoch=20, show_metric=True, run_id="embedding_model")


def build_nce_model(num_words, num_docs, doc_embedding_size=doc_embedding_size, word_embedding_size=word_embedding_size):
    X1 = tflearn.input_data(shape=[None, 1])
    X2 = tflearn.input_data(shape=[None, 3])
    
    Y = tf.placeholder(tf.float32, [None, 1])

    d1, = tflearn.embedding(X1, input_dim=num_docs, output_dim=doc_embedding_size)
    w1, w2, w3 = tflearn.embedding(X2, input_dim=num_words, output_dim=word_embedding_size)

    embedding_layer = tflearn.merge([d1, w1, w2, w3], mode='concat')

    num_classes = num_words
    dim = doc_embedding_size + 3*word_embedding_size
        
    with tf.variable_scope("NCELoss"):
        weights = tflearn.variables.variable('W', [num_classes, dim])
        biases  = tflearn.variables.variable('b', [num_classes])

        batch_loss = tf.nn.nce_loss(weights, biases, embedding_layer, Y, num_sampled=100, num_classes=num_classes)
        loss = tf.reduce_mean(batch_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    
    trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer,
                          metric=None, batch_size=32)

    trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=0, checkpoint_path='nce_test_model')
    return trainer, X1, X2, Y


def train_nce_model(trainer, X1, X2, Y, docs, contexts, targets, num_words):
	X1_data = docs[:,np.newaxis]
	X2_data = contexts
	Y_data  = targets[:,np.newaxis]
	trainer.fit(feed_dicts={X1: X1_data, X2: X2_data, Y: Y_data}, n_epoch=20, show_metric=False, run_id="nce_test_model")


if __name__ == '__main__':
	tweet_df, vocab = load_data()
	embedding_index_map = tweet_df[['id', 'text']]
	embedding_index_map.to_pickle('embedding_index_map.pkl')
	docs, contexts, targets = prepare_contexts(tweet_df)

	# model = build_model(num_words=len(vocab), num_docs=len(tweet_df))
	# train_model(model, docs, contexts, targets, num_words=len(vocab))
	# model.save('embedding_model')

	trainer, X1, X2, Y = build_nce_model(num_words=len(vocab), num_docs=len(tweet_df))
	train_nce_model(trainer, X1, X2, Y, docs, contexts, targets, num_words=len(vocab))
	trainer.save('nce_test_model')
