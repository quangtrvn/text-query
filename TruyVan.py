import os
import pickle
import sys
import HamHoTro as hht
import TienXuLyVanBan as txl
from collections import Counter
import math


doc_file=os.path.join(os.getcwd(),'data','docs.pickle')
inverted_index_file =os.path.join(os.getcwd(),'data','inverted_index.pickle')
stopwords_file=os.path.join(os.getcwd(),'resources','stopwords.txt')

d = open(doc_file,'rb')
docs = pickle.load(d)
i=open(inverted_index_file,'rb')
inverted_index= pickle.load(i)
stopwords=hht.get_stopwords(stopwords_file)

dictionary=set(inverted_index.keys())
#truy vấn
print('Nhập câu truy vấn')
query=input()
print('\n')
query=txl.preprocess_text(query,stopwords)
query=[word for word in query if word in dictionary]
query=Counter(query)

for word,value in query.items():
    query[word]=inverted_index[word]['idf'] * (value/len(query))
scores = [[i, 0] for i in range(len(docs))]
for word, value in query.items():
    for doc in inverted_index[word]['postings_list']:
        index, weight = doc
        scores[index][1] += value * weight

scores.sort(key=lambda doc: doc[1], reverse=True)

print('----- Results ------ ')
for index, score in enumerate(scores):
    if score[1] == 0:
        break
    else: print('{}. {} - {}'.format(index + 1, docs[score[0]], score[1]))
