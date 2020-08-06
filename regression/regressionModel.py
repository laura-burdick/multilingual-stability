# Laura Burdick (lburdick@umich.edu)
# Train regression model

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# SET THESE VARIABLES

# Location of file with average stability for each language (generated from Get Average Stabilities.ipynb)
# File should have the format of a csv file with columns "language" and
# "averageStability", where language is the Bible code for the language
# and averageStability is the average stability of that language
average_stabilities_path = 'data/average_stabilities_allLanguages.csv'

# Location of WALS features for each language for regression model (generated by Get Good WALS Values.ipynb)
# For each language, should be formatted as a pickle file with name
# {wals_features_path}{language}.pkl, where pickle file contains a list
# of WALS feature values for that language
wals_features_path = '~/embedding-spaces/multilingual_thesis/regression/data/allLanguages_language_features_small_wals_'

# Location to save each of the 1000 bootstrapped regression models
# Will be saved as {regression_model_path}_{bootstrap_iteration}.pkl, where each pickle file is a pickled sklearn Ridge Regression model
regression_model_path = '~/embedding-spaces/multilingual_thesis/regression/data/allLanguages_regression_model_averageLanguageStability_bootstrapping_'

# Location to save R^2 scores for each of the 1000 bootstrapped regression models
# Will be formatted as a pickle file, where the pickle is a list of 1000 float scores
regression_scores_path = '~/embedding-spaces/multilingual_thesis/regression/data/allLanguages_regression_model_averageLanguageStability_bootstrapping_scores.pkl'

# Set of languages to use in regression model (can adjust if needed, or leave the same)
languages = ['eng', 'rus', 'fin', 'hun', 'spa', 'tur', 'ind', 'mnd', 'jpn', 'kor', 'prs', 'hin', 'vie', 'heb', 'may', 'tha', 'lav', 'lat', 'hmo', 'cmn', 'pol', 'som', 'bul', 'ita', 'lit', 'swe', 'hat', 'nor', 'poh', 'est', 'mam', 'por', 'ukr', 'ben', 'che', 'lnd', 'mad']

print('Reading in all features...')
all_features = [] # All features for regression model
all_target = [] # All predicted output for regression model
average_stabilities = pd.read_csv(average_stabilities_path)

# Load language features and target output
for language in tqdm(languages):
	with open(wals_feature_path+language+'.pkl','rb') as pickleFile:
		if len(all_features) == 0:
			all_features = [pickle.load(pickleFile)]
		else:
			features = [pickle.load(pickleFile)]
			all_features += features
	all_target.append(list(average_stabilities.loc[average_stabilities.language==language]['averageStability'])[0]) # Target output = average stability of language

scores = [] # List of 1000 bootstrapped R2 scores
for iteration in range(0, 1000): # Bootstrap 1000 iterations
	print('iteration',iteration)
	indices = np.random.choice([i for i in range(len(all_features))],size=len(all_features),replace=True) # Randomly choose input features

	model = Ridge(random_state=42,alpha=10) # Ridge regression model

	print('Fitting model...')
	train_features = [all_features[i] for i in indices]
	target_features = [all_target[i] for i in indices]
	model.fit(train_features,target_features)
	
	print('Scoring model...')
	r2_score = model.score(train_features,target_features)
	print('r2_score',r2_score)
	scores.append(r2_score)

	print('Saving model...')
	with open(regression_model_path+str(iteration)+'.pkl','wb') as pickleFile:
		pickle.dump(model,pickleFile)
with open(regression_scores_path,'wb') as pickleFile:
	pickle.dump(scores,pickleFile)