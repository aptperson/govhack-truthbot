import tweepy
import threading
import collections
import re
import time
import sqlite3


consumer_key = 'sS25OzYp60Ghi81D5Jwx8KFiC'
consumer_secret = 'Z4wTkRCs2Yx31r6YWtsHbs1GEE08NChHGWndDAsKNyCJ7I4uVT'

access_token = '758106823786242048-qZK96DAGsHohztTieWyWHsJsRBBzJnf'
access_token_secret = 'AvqkYvAnPpoLqSgLgWrNAfqEPHzfrYIlLZZBDlgbd5zq5'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
con = sqlite3.connect('tweets.db')


def tweet_is_relevant(tweet):
	return re.search('muslim|islam|economy|jobs?', tweet.text, flags=re.IGNORECASE)


def process_new_tweets():
	# tweets = api.home_timeline()
	tweets = api.user_timeline(screen_name='corybernardi')

	for tweet in tweets:
		cur = con.execute('SELECT * FROM read_tweets where tweet_id = ?', (tweet.id,))
		if cur.fetchone() is not None:
			continue

		if tweet_is_relevant(tweet):
			print(tweet.text)

		with con:
			con.execute('INSERT INTO read_tweets VALUES (?)', (tweet.id,))


def main():
	con.execute("CREATE TABLE IF NOT EXISTS read_tweets (tweet_id)")

	while True:
		process_new_tweets()
		time.sleep(10)


if __name__ == "__main__":
    main()
