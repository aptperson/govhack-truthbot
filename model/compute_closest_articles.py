import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
import train

num_matches = 20


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


similarity_data = []

for doc_idx in tqdm(tweet_data.index):
    doc_vector = doc_embedding_norm[doc_idx]

    similarity = np.sum(doc_vector * news_embedding, axis=1)
    similarity_sort_idx = np.argsort(similarity)[::-1]

    articles = [(similarity[i], news_data.iloc[i]) for i in similarity_sort_idx[:num_matches]]
    similarity_data.append((tweet_data.loc[doc_idx], articles))

# with open('similarity_data.pkl', 'wb') as f:
#     pickle.dump(similarity_data, f)

tweets_by_screen_name = {}
for tweet, articles_and_sim in similarity_data:
    screen_name = tweet.twitter_screen_name
    tweets_by_screen_name[screen_name] = tweets_by_screen_name.get(tweet.twitter_screen_name,[])
    tweets_by_screen_name[screen_name].append({
        'text': tweet.text,
        'articles': [{
            'score': sim,
            'headline': article.headline,
            'article_url': article.article_url,
            } for sim, article in articles_and_sim
        ]
    })

with open('../tweets_with_data.pkl', 'wb') as f:
    pickle.dump(tweets_by_screen_name, f)
