import tweepy
import itertools
import csv

max_tweet_count = 200

def get_tweets_by_page(api, screen_name):
	tweets = api.user_timeline(screen_name=screen_name, count=max_tweet_count)
	yield tweets
	while len(tweets) > 0:
		tweets = api.user_timeline(screen_name=screen_name, count=max_tweet_count, max_id=tweets.max_id)
		yield tweets


def get_tweets(api, screen_name):
	return itertools.chain.from_iterable(get_tweets_by_page(api, screen_name))


if __name__ == "__main__":
	import sys

	consumer_key = 'sS25OzYp60Ghi81D5Jwx8KFiC'
	consumer_secret = 'Z4wTkRCs2Yx31r6YWtsHbs1GEE08NChHGWndDAsKNyCJ7I4uVT'

	access_token = '758106823786242048-qZK96DAGsHohztTieWyWHsJsRBBzJnf'
	access_token_secret = 'AvqkYvAnPpoLqSgLgWrNAfqEPHzfrYIlLZZBDlgbd5zq5'

	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)

	api = tweepy.API(auth)
	screen_name = sys.argv[1]

	tweets = get_tweets(api, screen_name)
	with open('%s_tweets.csv' % screen_name, 'w') as f:
		writer = csv.writer(f)
		writer.writerow(['id', 'text'])
		for tweet in tweets:
			writer.writerow([tweet.id, tweet.text])
