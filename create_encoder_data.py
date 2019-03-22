# run this code to create data for the encoder
from wordExtract import WordExtract as ext
import os
from random import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


keep =None

#### YOU MAY NEED TO CHENGE  root_dir TO POINT TO REDDIT DIRECTORY ###
root_dir = "/Users/apple/Desktop/q2_course/cs272/finalProject/reddit-dataset"

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
            sz = len(obj.meta)
            comments.extend(obj.text[:keep])
            metaLabels.extend(obj.meta[:keep])
            subLabels.extend(obj.sub[:keep])
            
meta = sorted(meta)
sub = sorted(sub)

meta2id = {}
sub2id = {}
for i,c in enumerate(meta):
    meta2id[c] = i
for i,c in enumerate(sub):
    sub2id[c] = i


stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

link = set(['com','png','gif','imgur','jpg','http','https'])
# filter out stop word / stem / special word
fil_comment, fil_metaLabels, fil_subLabels = [], [], []
for i in range(len(comments)):
    #tokens = word_tokenize(comments[i])
    tokens = comments[i].split()

    words=[]

    for j in range(len(tokens)):
        if any([l in tokens[j] for l in link]):
            continue
        elif tokens[j-1] in ['http','https']:
            continue
        elif len(tokens[j]) > 15:
            continue
        words.append(tokens[j])

    words = [porter.stem(w) for w in words if not w in stop_words]
    comments[i] = ' '.join(words)


    #words = [w for w in tokens if not w in stop_words]
    #stemmed = [porter.stem(word) for word in words]
    if not comments[i]:
        continue
    else:
        fil_comment.append(comments[i])
        fil_metaLabels.append(metaLabels[i])
        fil_subLabels.append(subLabels[i])
del comments, metaLabels, subLabels
print("Data Length is: ", len(fil_subLabels))

# tf-idf    
tfidf = TfidfVectorizer(norm=None)
tfidf_comment = tfidf.fit_transform(fil_comment)
index_value={i[1]:i[0] for i in tfidf.vocabulary_.items()}
fully_indexed = []
for row in tfidf_comment:
    fully_indexed.append({index_value[column]:value for (column,value) in zip(row.indices,row.data)})

clean_comment = []
for i in range(len(fil_comment)):
    tokens = fil_comment[i].split()
    strComment = []
    for word in tokens:
        if(word not in fully_indexed[i] or fully_indexed[i][word] >= 4.5):
            strComment.append(word)
    
    # avoid empty comment
    if not strComment:
        clean_comment.append(fil_comment[i])
    else: 
        clean_comment.append(' '.join(strComment))
        
del fil_comment


corpus = []
for i,c in enumerate(clean_comment):
    corpus.append([c, meta2id[fil_metaLabels[i]], sub2id[fil_subLabels[i]]])

'''
corpus = []
for i,c in enumerate(comments):
    corpus.append([c, meta2id[metaLabels[i]], sub2id[subLabels[i]]])
'''

#shuffle(corpus)

import pickle as pkl

with open('./data','wb') as f:
    pkl.dump(corpus,f)
