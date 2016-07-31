import pystache

with open('list.mustache') as f:
	list_template = f.read()

with open('friends.txt') as f:
	poli_list = [{'name': name.rstrip()} for name in f.readlines()]

list_page = pystache.render(list_template, {'poli': poli_list})

with open('list.html', 'w') as f:
	f.write(list_page)
