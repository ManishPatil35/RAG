# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:22:31 2024

@author: MANISH
"""

# scraper.py

import requests
from bs4 import BeautifulSoup

def scrape_wiki_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    content = ''
    for paragraph in soup.find_all('p'):
        content += paragraph.text
    
    return content

if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Luke_Skywalker"
    content = scrape_wiki_page(url)
    with open("luke_skywalker.txt", "w") as file:
        file.write(content)
