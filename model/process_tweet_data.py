import traceback
import glob
import pickle

from collections import Counter

import numpy as np
import pandas as pd

from tqdm import tqdm

import spacy

nlp = spacy.load('en')

unk_word = '<UNK>'
start_word = '<START>'

tweets = pd.concat(pd.read_csv(f, dtype={'id': np.int64}) for f in glob.glob('../tweet_data/*.csv'))

texts = tweets.text.values

c = Counter()
for text in tqdm(texts):
	try:
		doc = nlp(text)
		c.update(tok.text for tok in doc)
	except AssertionError:
		print('parse error:', text)
		# traceback.print_exc()

vocab = [start_word] + [w for w, n in c.most_common() if n > 50] + [unk_word]

word_to_vocab = {w: i for i, w in enumerate(vocab)}
unk_id = word_to_vocab[unk_word]

tweets['tokenized'] = tweets.text.apply(lambda t: [word_to_vocab.get(w, unk_id) for w in t.split()])


with open('tweet_vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

tweets.to_pickle('tweet_data_tokenized.pkl')
