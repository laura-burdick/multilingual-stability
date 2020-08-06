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
**stability/**: This folder contains code to calculate stability for both Wikipedia and the Bible corpora.
- Dependencies: [gensim](https://radimrehurek.com/gensim/), [GloVe](https://nlp.stanford.edu/projects/glove/), [faiss](https://github.com/facebookresearch/faiss), [sklearn](https://scikit-learn.org/), [numpy](https://numpy.org/), [tqdm](https://github.com/tqdm/tqdm), [pandas](https://pandas.pydata.org/)
- For the Bible, first you need to create five word2vec embedding spaces (with different random seeds) for each language. The script **w2v_bible.py** will do this. Before running this, you will need to have the Bible corpora downloaded, and you will need to set the variables at the top of the script.
- For Wikipedia, first you need to create five GloVe embedding spaces (downsampling without replacement) for each language. The script **trainGlove_wikipedia.sh** will do this. Before running this, you will need to have the GloVe code and Wikipedia corpora downloaded, and you will need to create five downsamples for each Wikipedia language (more information in script). You will also need to set the variables at the top of the script.
- Second, for each language in both the Bible and Wikipedia, you need to precalculate the five nearest neighbors for every word (this speeds up the stability calculation drastically). The scripts **precalculateNearestNeighbors_bible.py** and **precalculateNearestNeighbors_wikipedia.py** do this. For both scripts, you will need to set the variables at the top of the scripts.
- Finally, you can calculate stability for each language in both the Bible and Wikipedia, using **stability_bible.py** and **stability_wikipedia.py**. For both scripts, you will need to set the variables at the top of the scripts.

## Acknowledgements
This material is based in part upon work supported by the National Science Foundation (NSF \#1344257), the Defense Advanced Research Projects Agency (DARPA) AIDA program under grant \#FA8750-18-2-0019, and the Michigan Institute for Data Science (MIDAS). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NSF, DARPA, or MIDAS.
