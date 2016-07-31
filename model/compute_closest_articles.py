import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

import train

doc_data = pd.read_pickle('data_tokenized.pkl')

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
    vocab = np.array(vocab)

trainer, X1, X2, Y = train.build_nce_model(num_words = len(vocab), num_docs = len(doc_data))
trainer.restore('embedding_model_nce-51230')

e1, = trainer.graph.get_collection('layer_variables/Embedding/')
e2, = trainer.graph.get_collection('layer_variables/Embedding_1/')

doc_embedding  = e1.eval(session=trainer.session)
word_embedding = e2.eval(session=trainer.session)

doc_embedding_norm = doc_embedding / np.linalg.norm(doc_embedding, axis=1, keepdims=True)

news_data = doc_data[doc_data.document_kind == 'news']
news_embedding = doc_embedding_norm[news_data.index]

tweet_data = doc_data[doc_data.document_kind == 'twitter']

for doc_idx in tweet_data.index:
    doc = doc_embedding_norm[doc_idx]
    print('---------------------------')
    print(doc_data.iloc[doc_idx].text)
    print('---------------------------')

    dist = np.sum(doc * news_embedding, axis=1)
    dist_sort_idx = np.argsort(dist)[::-1]
    for i in dist_sort_idx[:25]:
        t = news_data.iloc[i].headline
        d = dist[i]
        print('---', np.round(d, 2))
        print(t)

    print('\n\n\n')
