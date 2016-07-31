import os
import pandas
import pystache
import pickle

with open('list.html.mustache') as f:
    list_template = f.read()

with open('poli.html.mustache') as f:
    poli_template = f.read()

# with open('friends.txt') as f:
#   poli_list = [{'name': name.rstrip(), 'url': 'poli/%s.html' % name.rstrip()} for name in f.readlines()]

with open('friends_info.pkl', 'rb') as f:
    friends_info = pickle.load(f)

with open('tweets_with_data.pkl', 'rb') as f:
    tweets_with_data = pickle.load(f)

for tweets in tweets_with_data.values():
    for tweet in tweets:
        tweet['articles'] = tweet['articles'][:5]
        tweet['top_article'] = tweet['articles'][0]
        for article in tweet['articles']:
            article['score'] = '%.2f' % article['score']


def add_url(poli):
    poli['url'] = 'poli/%s.html' % poli['screen_name']
    return poli

poli_list = [add_url(poli) for poli in friends_info]

list_page = pystache.render(list_template, {'polis': poli_list})

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

ensure_dir('site')
ensure_dir('site/poli')

with open('site/list.html', 'w') as f:
    f.write(list_page)

for poli in poli_list:
    with open('site/poli/%s.html' % poli['screen_name'], 'w') as f:
        poli_page = pystache.render(poli_template, {
            'poli': poli,
            'tweets': tweets_with_data.get(poli['screen_name'], [])
        })
        f.write(poli_page)
