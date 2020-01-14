
#!/usr/bin/env python
# coding: utf-8

from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from urllib.parse import urlparse
from urllib.error import HTTPError
from urllib.error import URLError
import re
import pandas as pd
import os
import requests
from requests.exceptions import HTTPError, Timeout,TooManyRedirects, RequestException
import pickle



companies = pd.read_csv('./FW__Innovative_company_urls/allInnovData_scraped_overview.csv', sep=';')
companies.head()

urls = companies['URL']


df = companies.copy()
df['status'] = 0


class crawler:
  visited = set()
  base_url = ''
  tags = ['h1','h2','h3','h4','h5','h6', 'p']
  pattern = ''
  word_count = 0
  max_count = 20000 #max word count for each document
  time_out = 2  #time out in seconds
  category = ''
  beid = ''

  def __init__(self):
      return
      

  def crawl(self, url, category, beid):   
    self.config(url)
    bs = self.visit('')      
    if bs:
        if bs.find('html') is None:
            return False
        lang = bs.find('html').get('lang')
        if lang is None:
            lang = 'Unknown'
        path = os.path.join(download_folder, category, lang, beid)
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.mkdir(folder)
        with open(path, 'w+') as file:
          paths = self.find_links(bs)
          self.write_to_file(bs, file)
          for path in paths:
            if self.word_count >= self.max_count:
              print(path, self.word_count)
              break
            bs = self.visit(path)
            if bs:
              #print(path, self.word_count)
              self.write_to_file(bs, file)
    else:
      return False

    
  def config(self, base_url):
    self.base_url = base_url
    self.visited.clear()
    self.visited.add('')
    self.word_count = 0
    self.pattern = re.compile(r'(^{}([^#]+))|(^\/.+)'.format(base_url)) #extract both absolute and relative url

  def visit(self, relative_url):
    try:
      url = urljoin(self.base_url, relative_url)
      print('visiting {}'.format(url))
      response = requests.get(url, timeout=self.time_out)
      response.raise_for_status()
      print(f'status: {response.status_code}')
      return BeautifulSoup(response.content, 'lxml')  
    except RequestException:
        print(RequestException)
        return False

  def find_links(self, bs):
    paths = set()
    for a in bs.find_all('a'):
      if a.has_attr('href') and self.pattern.match(a['href']):
        path = urlparse(a['href']).path
        if path != '' and '.' not in path:
          if len(path)>=1 and path[-1] == '/':
            path = path[:-1]
          if path not in paths:
              paths.add(path)
    return paths

  def write_to_file(self, bs, file):

    for tag in bs.find_all(self.tags):
      self.word_count += len(re.split('\W', tag.get_text()))
      file.write(tag.get_text().strip())
      file.write('\n')
      

if __name__ == "__main__":
    with open('./scraped_set', 'rb') as f:
        s = pickle.load(f)
    c_ = crawler()
    download_folder = '/Users/septem/Downloads/Companies'
    if not os.path.exists(download_folder):
      os.makedirs(download_folder)
    for i in range(len(companies)):
      c = companies.iloc[i, ]
      id = c['BEID']
      if id not in s:
          url = c['URL']
          if url == '0':
              continue
          beid = str(c['BEID']) + '.txt'
          if (c['Innov'] == 1):
            category = 'innov'
          else:
            category = 'traditional'
          c_.crawl(url, category, beid)




#url = 'https://www.te.com/global-en/home.html'
#url1 = 'https://deruijtermeubel.nl/'
#c = crawler()
#file = '/Users/septem/Downloads/Companies/test.txt'
#with open(file, 'w+') as f:
#  c.crawl(url1, f)






