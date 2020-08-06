# Laura Burdick (lburdick@umich.edu)
# Precalculate the five nearest neighbors for every word for every language in Wikipedia.

import faiss
import time
import tables as tb
import pickle
from sklearn.neighbors import BallTree
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm,trange
import sys
import pandas as pd

# SET THESE VARIABLES

# Location of the GloVe embedding spaces (created using trainGlove_wikipedia.sh)
# Files should be stored in the format {nicolai_path}{language}/downsampled_without_replacement_glove_{downsample_index}
nicolai_path = '/local/data/polyglot/'

# Location where output nearest neighbors will be stored
# Files will be stored in the format {output_path}{language}/downsampled_without_replacement_glove_nearestNeighbors_{downsample_index}.pkl, where the pickle file is a dictionary where the keys are words and the values are lists of ten nearest neighbors for each word.
output_path = nicolai_path

# Indices of downsamples to process (can adjust if needed, or leave the same)
seeds = [0,1,2,3,4]

# Languages to precalculate nearest neighbors for (can adjust if needed, or leave the same)
languages=['ar','bg','ca','cs','da','de','el','en','es','et','fa','fi','fr','he','hi','hr','hu','id','it','ja','ko','lt','lv','ms','nl','no','pl','pt','ro','ru','sk','sl','sr','sv','th','tl','tr','uk','vi','zh']

# Precalculate nearest neighbors for each language
for language in languages:
	print(language)

	# Precalculate nearest neighbors for each downsample
	for seed in seeds:
		print(seed)
		
		print('Load model...')
		with open(nicolai_path+language+'/downsampled_without_replacement_glove_'+str(seed)+'.txt','r',encoding='latin-1') as embeddingFile:
			embeddings = [i[:-1].split(' ') for i in embeddingFile.readlines()]
			embedding_words = [i[0] for i in embeddings]

		xb = np.matrix([[float(j) for j in i[1:]] for i in embeddings],dtype='float32') #database

		print('Normalizing vectors')
		for i in trange(len(xb)):
			xb[i] = normalize(xb[i])

		d = xb.shape[1] #dimension
		nb = xb.shape[0] #database size
		nq = len(embedding_words) #num queries
		print('d',d)
		print('nb',nb)
		print('nq',nq)

		print('Creating query matrix...')
		xq = xb[[i for i in range(len(embedding_words))],:]
		print(xq.shape)

		print('Building index...')
		faiss_index = faiss.IndexFlatL2(d)
		faiss_index.add(xb) 

		k = 11 #number of nearest neighbors

		print('Calculating nearest neighbors...')
		D, I = faiss_index.search(xq, k)
		
		nearestNeighbors = {}
		print('Recording nearest neighbors...')
		for i in tqdm(range(len(embedding_words))):
			word = embedding_words[i]
			nearestNeighbors[word] = [embedding_words[j] for j in I[i]][1:]
		
		#Save final
		print('Saving nearest neighbors...')
		with open(output_path+language+'/downsampled_without_replacement_glove_nearestNeighbors_'+str(seed)+'.pkl','wb') as pickleFile:
			pickle.dump(nearestNeighbors,pickleFile)
