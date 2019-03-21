from gensim.models.doc2vec import Doc2Vec
import pickle as pkl

d2v = Doc2Vec.load('./d2v.model')
thread_path = "../threads_clean"
dialogs = []
with open(thread_path,'rb') as f:
	threads = pkl.load(f)
for t in threads:
	docvec = model.infer_vector(t.split(),steps=20,alpha=0.025)
	sims = model.docvecs.most_similar([docvec], topn=2)
	dialogs.append((t,sims[0][0]))
	print(f'similarity={sims[0][1]}')
	
with open('./dialog_data','wb') as f:
	pkl.dump(dialogs, f, protocol=pkl.HIGHEST_PROTOCOL)