# Analyzing the Surprising Variability in Word Embedding Stability Across Languages
Laura (Wendlandt) Burdick, Jonathan K. Kummerfeld, Rada Mihalcea

Language and Information Technologies (LIT)

University of Michigan

## Introduction
The code in this repository was used in Chapter 4 of Laura Burdick's Ph.D. thesis. I have tried to document it well, but at the end of the day, it is research code, so if you have any problems using it, please get in touch with Laura Burdick (lburdick@umich.edu).

## Citation Information
If you use this code, please cite the following paper:
```
(citation forthcoming)
```

## Code Included
**Complete Bibles.ipynb**: Contains code to figure out which Bible translations have 75% of the English KJV Bible.
- Dependencies: [Jupyter notebook](https://jupyter.org/), [tqdm](https://github.com/tqdm/tqdm)
- To run this, you will need to download the Bible corpora and set the variables at the top of the notebook.
- **all_bible_texts.txt** is also included as a helper file. This is a list of all the Bible translations.
**stability/**: This folder contains code to calculate stability for both Wikipedia and the Bible corpora.
- Dependencies: [gensim](https://radimrehurek.com/gensim/), [GloVe](https://nlp.stanford.edu/projects/glove/), [faiss](https://github.com/facebookresearch/faiss), [sklearn](https://scikit-learn.org/), [numpy](https://numpy.org/), [tqdm](https://github.com/tqdm/tqdm), [pandas](https://pandas.pydata.org/)
- For the Bible, first you need to create five word2vec embedding spaces (with different random seeds) for each language. The script **w2v_bible.py** will do this. Before running this, you will need to have the Bible corpora downloaded, and you will need to set the variables at the top of the script.
- For Wikipedia, first you need to create five GloVe embedding spaces (downsampling without replacement) for each language. The script **trainGlove_wikipedia.sh** will do this. Before running this, you will need to have the GloVe code and Wikipedia corpora downloaded, and you will need to create five downsamples for each Wikipedia language (more information in script). You will also need to set the variables at the top of the script.
- Second, for each language in both the Bible and Wikipedia, you need to precalculate the five nearest neighbors for every word (this speeds up the stability calculation drastically). The scripts **precalculateNearestNeighbors_bible.py** and **precalculateNearestNeighbors_wikipedia.py** do this. For both scripts, you will need to set the variables at the top of the scripts.
- Finally, you can calculate stability for each language in both the Bible and Wikipedia, using **stability_bible.py** and **stability_wikipedia.py**. For both scripts, you will need to set the variables at the top of the scripts.

**regression/**: This folder contains code to run the regression model described in the paper.
- Dependencies: [Jupyter notebook](https://jupyter.org/), [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [sklearn](https://scikit-learn.org/), [tqdm](https://github.com/tqdm/tqdm)
- First, you need to get the average stability for each language, found in **Get Average Stabilities.ipynb**. You will need to have already run stability for both the Bible and Wikipedia, as well as set the variables at the top of the notebook.
- Second, we need to run a number of scripts to get the correct input for the regression model. Run **Getting WALS Data.ipynb** to correctly format the raw WALS data. Before running this, you will need to download WALS, as well as set the variables at the top of the notebook. Run **Making Wals Binary.ipynb** to turn the WALS data into a set of binary features. You will need to set the variables at the top of the notebook. Run **Get Good WALS values.ipynb** to filter WALS values and languages for the final regression model. You will need to set the variables at the top of the notebook.
- Now, we can run the regression model using **regressionModel.py**. Make sure to set the variables at the top of the script.
- **crossValidation.py** contains code to run leave-one-out cross-validation for ridge regression models with different regularization values. Make sure to set the variables at the top of the script.
- A few data files are included that are referenced in the various scripts. **correlations.csv** contains Spearman's correlations between all the relevant WALS properties. It also includes a manual grouping of highly correlated properties. **multilingual_corpora.csv** contains a partial mapping between the Wikipedia language identifiers, the Bible language identifiers (ISO 639-3 Code), and the full name of the language. This was compiled manually.

## Acknowledgements
This material is based in part upon work supported by the National Science Foundation (NSF \#1344257), the Defense Advanced Research Projects Agency (DARPA) AIDA program under grant \#FA8750-18-2-0019, and the Michigan Institute for Data Science (MIDAS). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NSF, DARPA, or MIDAS.
