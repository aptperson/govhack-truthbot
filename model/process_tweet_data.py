import glob
import pickle

from collections import Counter

import pandas as pd
import numpy as np

tweets = pd.concat(pd.read_csv(f, dtype={'id': np.int64}) for f in glob.glob('../tweet_data/*.csv'))

texts = tweets.text.values

c = Counter(tok for text in texts for tok in text.split())

vocab = [w for w, n in c.most_common() if n > 50]
vocab.append('UNK')

unk_id = len(vocab) - 1
word_to_vocab = {w: i for i, w in enumerate(vocab)}

tweets['tokenized'] = tweets.text.apply(lambda t: [word_to_vocab.get(w, unk_id) for w in t.split()])


with open('tweet_vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

tweets.to_pickle('tweet_data_tokenized.pkl')
