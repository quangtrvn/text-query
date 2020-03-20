import re
from nltk.stem.porter import PorterStemmer
import os
def remove_nonwords(text):
    nonwords=re.compile(r"[^a-z']")
    processed_text=re.sub(nonwords,' ',text)
    return processed_text.strip()

def remove_stopwords(text,stopwords):
    words=[]
    for word in text.split():
        if word not in stopwords:
           words.append(word)
    return words

#trả về từ khởi nguồi
def stem_word(words):
    stemmer=PorterStemmer()
    stemmed_words=[stemmer.stem(word) for word in words]
    return stemmed_words

def preprocess_text(text,stopwords):
    processed_text=remove_nonwords(text)
    words=remove_stopwords(processed_text,stopwords)
    stemmed_words=stem_word(words)
    return stemmed_words
data=os.path.join(os.getcwd(),'doc.txt')
data=open(data,mode='r')
data=data.read()
print(remove_nonwords(data))
