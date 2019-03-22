from wordExtract import WordExtract as ext
import os
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

#sorts subreddit and met subreddit alphabetically since python 
#dictionaries change the order when you iterate
meta = sorted(meta)
sub = sorted(sub)

meta2id = {}
sub2id = {}
for i,c in enumerate(meta):
    meta2id[c] = i
for i,c in enumerate(sub):
    sub2id[c] = i


# print(meta2id)
# print()
# print(sub2id)
# print()


#200 is about 28 words of 7 characters each
#300 is about 43 words of 7 characters each
length_limit = 30


corpus = []
for i, comment in enumerate(comments):

    #splits up the words
    split_comment = comment.split(" ")

    #splits up comments if longer than limit
    # print(str(i)+": "+str(comment))

    if len(split_comment) > length_limit:
        start = length_limit
        end = length_limit*2
        #rewrites the original comment
        # print("Breaking up: "+str(split_comment[:start]))
        corpus.append([' '.join(split_comment[:start]), meta2id[metaLabels[i]], sub2id[subLabels[i]]])

        #appends rest of the comment
        while end < len(split_comment):
            # print("Breaking up: "+str(split_comment[start:end]))
            corpus.append([' '.join(split_comment[start:end]), meta2id[metaLabels[i]], sub2id[subLabels[i]]])
            start = end
            end += length_limit

        #inserts the rest of the comment
        if start<len(split_comment):
            # print("Breaking up: "+str(split_comment[start:]))
            corpus.append([' '.join(split_comment[start:]), meta2id[metaLabels[i]], sub2id[subLabels[i]]])
        # input()
    else:
        corpus.append([comment, meta2id[metaLabels[i]], sub2id[subLabels[i]]])

    # print(corpus)
    # input()

import pickle as pkl
'''
with open('./data','wb') as f:
    pkl.dump(corpus,f)
    
'''