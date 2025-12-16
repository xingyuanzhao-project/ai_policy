import json
import random
from bs4 import BeautifulSoup

with open('data/ncsl/ncsl_bills.jsonl', encoding='utf-8') as f:
    records = [json.loads(l) for l in f]

random.seed(99)
samples = random.sample(records, 5)

for i, rec in enumerate(samples):
    print(f'=== SAMPLE {i+1}: {rec["bill_id"]} ===')
    soup = BeautifulSoup(rec['html'], 'html.parser')
    
    text_id = soup.find('table', id='text-identifier')
    doc_body = soup.find('div', class_='documentBody')
    div_head = soup.find('div', class_='head')
    div_title = soup.find('div', class_='title')
    div_text = soup.find('div', class_='text')
    td_key = soup.find('td', class_='key')
    td_labels = soup.find_all('td', class_='label')
    
    print(f'  table#text-identifier: {"YES" if text_id else "NO"}')
    print(f'  td.key: {td_key.get_text(strip=True)[:40] if td_key else "NO"}')
    print(f'  td.label count: {len(td_labels)}')
    print(f'  div.documentBody: {"YES" if doc_body else "NO"}')
    print(f'  div.head: {"YES" if div_head else "NO"}')
    print(f'  div.title: {"YES" if div_title else "NO"}')
    print(f'  div.text: {"YES" if div_text else "NO"}')
    if div_text:
        text_len = len(div_text.get_text())
        print(f'  div.text length: {text_len} chars')
    print()
