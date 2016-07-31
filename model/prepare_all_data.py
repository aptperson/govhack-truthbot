import traceback
import re
import glob
import pickle

from collections import Counter

import numpy as np
import pandas as pd

from tqdm import tqdm

import spacy

nlp = spacy.load('en', parser=False, tagger=False)

unk_word = '<UNK>'
start_word = '<START>'

vocab_threshold = 20

data = pd.read_pickle('../alldata.pkl')
texts = data.text.values

c = Counter()
for text in tqdm(texts):
    try:
        text = re.sub(r'\.+', '.', text)
        text = text.replace('\n', ' ')
        doc = nlp(text, tag=False)
        c.update(tok.text for tok in doc)
    except AssertionError:
        print('parse error:', text)
        # traceback.print_exc()

vocab = [start_word] + [w for w, n in c.most_common() if n > vocab_threshold] + [unk_word]

word_to_vocab = {w: i for i, w in enumerate(vocab)}
unk_id = word_to_vocab[unk_word]

data['tokenized'] = data.text.apply(lambda t: [word_to_vocab.get(w, unk_id) for w in t.split()])

with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

data.to_pickle('data_tokenized.pkl')
