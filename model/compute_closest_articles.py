import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

import train

embedding_index_map = pd.read_pickle('embedding_index_map.pkl')

with open('tweet_vocab.pkl', 'rb') as f:
	vocab = pickle.load(f)
	vocab = np.array(vocab)

trainer, X1, X2, Y = train.build_nce_model(num_words = len(vocab), num_docs = len(embedding_index_map))
trainer.restore('nce_test_model-46674')

e1, = trainer.graph.get_collection('layer_variables/Embedding/')
e2, = trainer.graph.get_collection('layer_variables/Embedding_1/')

doc_embedding  = e1.eval(session=trainer.session)
word_embedding = e2.eval(session=trainer.session)

doc_embedding_norm = doc_embedding / np.linalg.norm(doc_embedding, axis=1, keepdims=True)

# for doc_idx in range(len(embedding_index_map))
for doc_idx in range(10):
	doc = doc_embedding_norm[doc_idx]

	print('---------------------------')

	dist = np.sum(doc * doc_embedding_norm, axis=1)
	dist_sort_idx = np.argsort(dist)[::-1]
	for i in dist_sort_idx[:5]:
		t = embedding_index_map.iloc[i].text
		d = dist[i]
		print('---', np.round(d, 2))
		print(t)
