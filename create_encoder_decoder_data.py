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


#sorts subreddit and met subreddit alphabetically since python 
#dictionaries change the order when you iterate
meta = sorted(meta)
sub = sorted(sub)

meta2id = {}
sub2id = {}
for index,comment in enumerate(meta):
    meta2id[comment] = index
for index,comment in enumerate(sub):
    sub2id[comment] = index


# print(meta2id)
# print()
# print(sub2id)
# print()



corpus = []
for x in range(0, len(comments)-1):
    corpus.append([comments[x], comments[x+1]])
    # print("Comment: "+str(comments[x]))
    # print("Reply: "+str(comments[x+1]))
    # print(corpus)
    # input()

import pickle as pkl

with open('./encoder_decoder_data','wb') as f:
    pkl.dump(corpus,f)