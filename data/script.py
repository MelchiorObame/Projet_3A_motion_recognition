import urllib.request

print('Beginning file download with urllib2...')

url = 'http://i3.ytimg.com/vi/J---aiyznGQ/mqdefault.jpg'
urllib.request.urlretrieve(url, '/Users/scott/Downloads/cat.jpg')