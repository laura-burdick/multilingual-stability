# Laura Burdick (lburdick@umich.edu)
# Calculate stability for each language in Wikipedia.

import numpy as np
from sklearn.neighbors import BallTree
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import sys
from tqdm import tqdm,trange
import pandas as pd

# SET THESE VARIABLES

# Location of the nearest neighbors for each word (created using precalculateNearestNeighbors_wikipedia.py)
# Files should be stored in the format {output_path}{language}/downsampled_without_replacement_glove_nearestNeighbors_{downsample_index}.pkl, where the pickle file is a dictionary where the keys are words and the values are lists of ten nearest neighbors for each word.
nicolai_path = '/local/data/polyglot/'

# Location where output stability will be stored
# Files will be stored in the format {output_path}{language}/{language}_downsampled_without_replacement_glove_stability.csv, where the csv file has columns "word" and "stability", and the stability value is recorded for each word
output_path = nicolai_path

# Indices of downsamples to process (can adjust if needed, or leave the same)
seeds = [0,1,2,3,4]

# Languages to calculate stability for (can adjust if needed, or leave the same)
languages=['ar','bg','ca','cs','da','de','el','en','es','et','fa','fi','fr','he','hi','hr','hu','id','it','ja','ko','lt','lv','ms','nl','no','pl','pt','ro','ru','sk','sl','sr','sv','th','tl','tr','uk','vi','zh']

# Calculates the stability of a word in two sets of embedding spaces
# Assumes that you've already calculated the most similar words for the word
#
# @param word
#    The word to calculate stability for
# @param similar1
#    The list of nearest neighbors to word in the first set of embedding spaces
#    len(similar1) = # of embedding spaces in the first set
#    For each i, len(similar1[i]) = # of nearest neighbors to consider (same for each i)
# @param similar2
#    The list of nearest neighbors to word in the second set of embedding spaces
# @param same
#    Are the two lists of embedding spaces the same? (default = False)
#
# @returns a float, the average stability of the word across the two sets of spaces
#
def stability(word,similar1,similar2,same=False):
    if same and len(similar1) == 1:
        return len(similar1[0])
    
    sets1 = [set(a) for a in similar1]
    if not same:
        sets2 = [set(b) for b in similar2]
    else:
        sets2 = sets1
    
    avgOverlap = 0
    for i in range(len(similar1)):
        for j in range(len(similar2)):
            if not same or (same and i!=j):
                avgOverlap += len(sets1[i] & sets2[j])

    if same:
        avgOverlap /= (len(similar1)*len(similar2)-len(similar1))
    else:
        avgOverlap /= (len(similar1)*len(similar2))
    return avgOverlap

# Calculate stability for each language
for language in languages:
	print(language)
	
	print('Reading ten nearest neighbors...')
	nearest_neighbors = []
	words = set()
	for seed in seeds:
		print(seed)
		with open(nicolai_path+language+'/downsampled_without_replacement_glove_nearestNeighbors_'+str(seed)+'.pkl','rb') as pickleFile:
			nearest_neighbors.append(pickle.load(pickleFile))
			_words = set(nearest_neighbors[-1].keys())
			if len(words)==0:
				words = _words
			else:
				words = words.intersection(_words)
	words = list(words)

	print('Calculating stabilities...')
	stabilities = []
	for word in tqdm(words):
		most_similar = []
		for i in range(5):
			most_similar.append(nearest_neighbors[i][word])
		stabilities.append(stability(word,most_similar,most_similar,True))

	print('Writing output file...')
	df = pd.DataFrame(data={'word':words,'stability':stabilities})
	df.to_csv(output_path+language+'/'+language+'_downsampled_without_replacement_glove_stability.csv')
