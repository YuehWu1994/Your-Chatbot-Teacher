from gensim.models.doc2vec import LabeledSentence ,Doc2Vec, TaggedDocument, FAST_VERSION
import os
from os import listdir
from os.path import isfile, join
from wordExtract import WordExtract as ext
import pickle as pkl
from tqdm import tqdm
from sklearn import utils
import sys
assert FAST_VERSION > -1

docs = []
docsLabels = []
thread_path = "../threads_clean"
root_dir =  '/Users/william/Data/reddit-dataset-master'
# bad=0
# for _, _, fileList in os.walk(root_dir):
#     for f in fileList:
#         if f.endswith(".csv"):
#             print(f)
#             obj = ext(root_dir+'/'+f)
#             obj.extract()
#             txt = obj.text
#             for i in range(len(txt)):
#             	if any(d in txt[i] for d in ['http','https','jpg','gif','png']):
#             		bad+=1
#             		continue
#             	docs.append(txt[i].split())
#             	# docsLabels.append([sub[i]])

# print(f'filtered out {bad} comments')
# assert len(docs) == len(docsLabels)
# with open('_words','wb') as f:
# 	pkl.dump(docs,f)
# with open('_labels','wb') as f:
# 	pkl.dump(docsLabels,f)
# del docs 
# del docsLabels
with open('./_words','rb') as f:
	docs = pkl.load(f)
with open('./_labels', 'rb') as f:
	docsLabels = pkl.load(f)
print ('loaded')
docsLabels = [[i] for i in range(len(docs))]
assert len(docs) == len(docsLabels)
print(docs[0])
print(docsLabels[0])
sentences = [TaggedDocument(words=docs[i], tags=docsLabels[i]) for i in range(len(docs))]
del docs, docsLabels
model = Doc2Vec(dm=1 ,vector_size=150, min_count=5, sample=1e-4, window=10, negative=5 ,alpha=0.025, hs=0, workers=4)

model.build_vocab([x for x in tqdm(sentences)])
for epoch in tqdm(range(20)):
	model.train(sentences, total_examples=model.corpus_count, epochs=1)
	model.alpha -= 0.002
	model.min_alpha = model.alpha

model.save("d2v.model")
print('model complete')




