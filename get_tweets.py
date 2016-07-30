import tweepy
import itertools
import csv

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

	tweets = tweepy.Cursor(api.user_timeline, screen_name=screen_name, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, count=200).items()
	with open('%s.csv' % screen_name, 'w') as f:
		writer = csv.writer(f)
		writer.writerow(['id', 'text'])
		for tweet in tweets:
			writer.writerow([tweet.id, tweet.text])
