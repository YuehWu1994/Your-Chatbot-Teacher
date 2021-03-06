# run this code to create data for the encoder
#

from wordExtract import WordExtract as ext
import os
keep =None

#### YOU MAY NEED TO CHENGE  root_dir TO POINT TO REDDIT DIRECTORY ###
root_dir =  os.getcwd() + "./reddit-dataset"

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

corpus = []
for i,c in enumerate(comments):
    corpus.append([c, meta2id[metaLabels[i]], sub2id[subLabels[i]]])

import pickle as pkl

with open('./data','wb') as f:
    pkl.dump(corpus,f)