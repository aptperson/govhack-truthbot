import os
import pandas
import pystache

with open('list.html.mustache') as f:
	list_template = f.read()

with open('poli.html.mustache') as f:
	poli_template = f.read()

with open('friends.txt') as f:
	poli_list = [{'name': name.rstrip(), 'url': 'poli/%s.html' % name.rstrip()} for name in f.readlines()]

list_page = pystache.render(list_template, {'polis': poli_list})

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

ensure_dir('site')
ensure_dir('site/poli')

with open('site/list.html', 'w') as f:
	f.write(list_page)

for poli in poli_list:
	with open('site/poli/%s.html' % poli['name'], 'w') as f:
		poli_page = pystache.render(poli_template, {'poli': poli})
		f.write(poli_page)
