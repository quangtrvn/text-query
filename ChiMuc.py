import os
import math
import HamHoTro as hht
import TienXuLyVanBan as txl
import pickle
from collections import Counter
import sys
print('beginning')
resources_path=os.path.join(os.getcwd(),'resources')
data_path=os.path.join(os.getcwd(),'data')
if not os.path.isdir(resources_path):
    print(':{} không có dữ liệu '.format(resources_path))
    sys.exit()
if not os.path.isdir(data_path):
    os.mkdir(data_path)

#lấy dữ liệu và file stopwords
dulieu_path=os.path.join(resources_path,'documents')
stopwords_file=os.path.join(resources_path,'stopwords.txt')

#tách và lấy các stopword
stopwords=hht.get_stopwords(stopwords_file)

#lấy các document
docs=hht.get_docs(dulieu_path)

tapvanban=[]
for doc in docs:
    f=open(doc, mode='r')
    text=f.read()
    words=txl.preprocess_text(text,stopwords)
    bag_of_words=Counter(words)
    tapvanban.append(bag_of_words)
idf_value=hht.idf(tapvanban)
for doc in tapvanban:
    hht.tf_idf(idf_value, doc)
inverted_index_value=hht.inverted_index(idf_value,tapvanban)
docs_file=os.path.join(data_path,'docs.pickle')
inverted_index_file=os.path.join(data_path,'inverted_index.pickle')
dictionary=os.path.join(data_path,'dictionary.txt')

#ghi dữ liệu
d=open(docs_file, 'wb')
pickle.dump(docs,d)

i=open(inverted_index_file, 'wb')
pickle.dump(inverted_index_value, i)

dic=open(dictionary,'w')
for word in idf_value.keys():
    dic.write(word+'\n')
print('done')
