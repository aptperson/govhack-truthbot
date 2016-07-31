import os
import glob

import numpy as np
import pandas as pd

n_tweets = 100

def load_twitter_df(f):
    screen_name = os.path.splitext(os.path.basename(f))[0]
    df = pd.read_csv(f, dtype={'id': str})[:n_tweets]
    return pd.DataFrame({
        'text': df.text,
        'document_kind': 'twitter',
        'tweet_id': df.id,
        'twitter_screen_name': screen_name
    })

twitter_data = pd.concat([load_twitter_df(f) for f in glob.glob('tweet_data/*.csv')])


explainers = pd.read_pickle('allexplainers.pickle')
explainers = explainers.dropna() 

explainer_data = pd.DataFrame({
    'text': explainers.content.apply(lambda s: s.decode("utf-8")),
    'document_kind': 'news',
    'article_url': explainers.link,
    'keywords': explainers.keywords,
    'headline': explainers.headline,
    'source': explainers.source
})

all_data = pd.concat([explainer_data, twitter_data])
all_data = all_data.reset_index(drop=True)
all_data.to_pickle('alldata.pkl')
