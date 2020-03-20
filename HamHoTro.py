import os
import math
from collections import defaultdict

#hàm lấy các stop word từ file có sẵn
def get_stopwords(stopwordsfile):
    f = open(stopwordsfile,'r')
    stopwords = set(f.read().split())
    return stopwords

#hàm lấy các văn bản từ thư mục
def get_docs(datapath):
    docs=[]
    for doc_file in os.listdir(datapath):
        docs.append(os.path.join(datapath,doc_file))
    return docs

#tính idf
def idf(folder):
    num_docs=len(folder)
    idf_Value=defaultdict(lambda :0)
    for doc in folder:
        for word in doc.keys():
            idf_Value[word] += 1
    for word,value in idf_Value.items():
        idf_Value[word] = math.log(num_docs/value)
    return idf_Value

def tf_idf(idf, doc):
    size=len(doc)
    for word,value in doc.items():
        doc[word] = idf[word]*(value/size)
#xây dựng chỉ mục ngược
def inverted_index(idf, docs):
    inverted_index_value = {}
    for word, value in idf.items():
        inverted_index_value[word] = {}
        inverted_index_value[word]['idf'] = value
        inverted_index_value[word]['postings_list'] = []
    for index, doc in enumerate(docs):
        for word, value in doc.items():
            inverted_index_value[word]['postings_list'].append([index,value])
    return inverted_index_value
