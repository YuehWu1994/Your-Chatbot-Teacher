# run this code to create data for the encoder
#

from wordExtract import WordExtract as ext
import os
from random import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer



keep =None

#### YOU MAY NEED TO CHENGE  root_dir TO POINT TO REDDIT DIRECTORY ###
root_dir = "/Users/kaku/Data/reddit-dataset-master/"

#os.chdir(root_dir)
comments, metaLabels, subLabels = [],[],[]
sub = set()
meta = set()
for _, _, fileList in os.walk(root_dir):
    for f in fileList:
        if f.endswith(".csv"):
            print(f)
            meta.add(f.split('.')[0].split('_')[0])
            sub.add(f.split('.')[0].split('_')[1])
            obj = ext(root_dir+'/'+f)
            obj.extract()
            comments.extend(obj.text[:keep])
            metaLabels.extend(obj.meta[:keep])
            subLabels.extend(obj.sub[:keep])

meta2id = {}
sub2id = {}
for i,c in enumerate(meta):
    meta2id[c] = i
for i,c in enumerate(sub):
    sub2id[c] = i
    
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

for i in range(len(comments)):
    if i%1000 ==0:
        print(i)
    tokens = word_tokenize(comments[i])
    # filter out stop word
    link = set(['com','png','gif','imgur','jpg','http','https'])
    words=[]

    for i in range(len(tokens)):
        if any([l in tokens[i] for l in link]):
            continue
        elif tokens[i-1] in ['http','https']:
            continue
        elif len(tokens[i]) > 15:
            continue
        words.append(tokens[i])

    words = [porter.stem(w) for w in words if not w in stop_words]
    comments[i] = ' '.join(words)
import pdb
pdb.set_trace

corpus = []
for i,c in enumerate(comments):
    corpus.append([c, meta2id[metaLabels[i]], sub2id[subLabels[i]]])


shuffle(corpus)
import pickle as pkl

with open('./data','wb') as f:
    pkl.dump(corpus,f)
