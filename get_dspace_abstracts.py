import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import re
import os

def download():
    id_list = get_ids()
    for id in tqdm(id_list):
        filename = 'datasets/dspace/%s.json' % id
        if not os.path.exists(filename):
            metadata = get_metadata_from_page(id)
            with open(filename, 'w') as f:
                json.dump(metadata, f)

def get_ids():
    ids = range(1431, 102338)
    # ids = range(1431, 1500)
    return ids[40000:]

def get_metadata_from_page(id):
    url = 'https://dspace.mit.edu/handle/1721.1/%d' % id
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    abstract = get_abstract(soup)
    author = get_author(soup)
    title = get_title(soup)
    dept = get_dept(soup)

    metadata = {'abstract': abstract, \
                'author': author, \
                'title': title, \
                'dept': dept}
    return metadata

def get_abstract(soup):
    class_name = 'simple-item-view-description'
    div_list = soup.findAll("div", {"class": class_name})
    if len(div_list) > 0:
        return div_list[0].text[11:].strip()

def get_title(soup):
    class_name = 'item-view-head'
    div_list = soup.findAll("h1")
    if len(div_list) > 0:
        return div_list[0].text.strip()


def get_author(soup):
    class_name = 'simple-item-view-authors'
    div_list = soup.findAll("div", {"class": class_name})
    if len(div_list) > 0:
        return div_list[0].text[9:].strip()

def get_dept(soup):
    class_name = 'simple-item-view-departments'
    div_list = soup.findAll("div", {"class": class_name})
    if len(div_list) > 0:
        return div_list[0].text[12:].strip()


if __name__ == '__main__':
    download()
