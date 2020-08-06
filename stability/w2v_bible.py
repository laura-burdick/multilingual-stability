# Laura Burdick (lburdick@umich.edu)
# Create five word2vec embedding spaces (with different random seeds) for each language of the Bible.

from collections import Counter
from gensim.models import word2vec
import sys
import pickle
import string

# SET THESE VARIABLES

# Location where the Bible text is stored
nicolai_path='/local/data/multilingual/bible/nicolai/text/'

# Location where output embedding spaces will be stored
# Files will be stored in the format {output_path}{language}_{word2vec_seed}.pkl, where the pickle file is a pickled gensim word2vec model
output_path = nicolai_path

# List of word2vec seeds to use (can adjust if needed, or leave the same)
seeds = [2518,2548,2590,29,401]

# List of Bible translations to create embedding spaces for (can adjust if needed, or leave the same)
# (The next line of code will remove the .txt ending from each language, so this list assumes that each language will have the .txt ending)
languages=['afr-x-bible-1953-v1.txt', 'aln-x-bible-aln-v1.txt', 'arb-ARBIBS.txt', 'arb-x-bible-arb-v1.txt', 'arz-x-bible-arz-v1.txt', 'ayr-AYMBSB.txt', 'ayr-x-bible-1997-v1.txt', 'ayr-x-bible-2011-v1.txt', 'azb-x-bible-azb-v1.txt', 'azj-AZEBSA.txt', 'bba-BBABSB.txt', 'bba-x-bible-bba-v1.txt', 'ben-x-bible-common-v1.txt', 'ben-x-bible-mussolmani-v1.txt', 'bqc-BQCSIM.txt', 'bqc-x-bible-bqc-v1.txt', 'bul-x-bible-bul-v1.txt', 'bul-x-bible-veren-v1.txt', 'cac-x-bible-ixtatan-v1.txt', 'cak-x-bible-central2003-v1.txt', 'ceb-x-bible-bugna2009-v1.txt', 'ceb-x-bible-bugna-v1.txt', 'ceb-x-bible-godsword-v1.txt', 'ceb-x-bible-pinadayag-v1.txt', 'ces-x-bible-ekumenicky-v1.txt', 'ces-x-bible-kralicka-v1.txt', 'che-CHEIBT.txt', 'cme-CNHBSM.txt', 'cmn-x-bible-sf_ncv-zefania-v1.txt', 'cnh-x-bible-cnh-v1.txt', 'crh-CRHIBT.txt', 'cym-x-bible-colloquial2013-v1.txt', 'cym-x-bible-morgan1804-v1.txt', 'dan-x-bible-1931-v1.txt', 'deu-x-bible-elberfelder1871-v1.txt', 'deu-x-bible-elberfelder1905-v1.txt', 'deu-x-bible-freebible-v1.txt', 'deu-x-bible-gruenewalder-v1.txt', 'deu-x-bible-luther1545letztehand-v1.txt', 'deu-x-bible-luther1545-v1.txt', 'deu-x-bible-luther1912-v1.txt', 'deu-x-bible-neue-v1.txt', 'deu-x-bible-pattloch-v1.txt', 'deu-x-bible-schlachter-v1.txt', 'deu-x-bible-tafelbibel-v1.txt', 'deu-x-bible-textbibel-v1.txt', 'deu-x-bible-zuercher-v1.txt', 'dyu-DYUWYI.txt', 'ell-x-bible-modern2009-v1.txt', 'eng-x-bible-darby-v1.txt', 'eng-x-bible-kingjames-v1.txt', 'eng-x-bible-literal-v1.txt', 'eng-x-bible-newsimplified-v1.txt', 'epo-x-bible-epo-v1.txt', 'fin-x-bible-1766-v1.txt', 'fin-x-bible-1933-v1.txt', 'fin-x-bible-1992-v1.txt', 'fra-x-bible-bonnet-v1.txt', 'fra-x-bible-crampon-v1.txt', 'fra-x-bible-darby-v1.txt', 'fra-x-bible-davidmartin-v1.txt', 'fra-x-bible-jerusalem2004-v1.txt', 'fra-x-bible-kingjames-v1.txt', 'fra-x-bible-louissegond-v1.txt', 'fra-x-bible-ostervald1867-v1.txt', 'fra-x-bible-paroledevie-v1.txt', 'fra-x-bible-perret-v1.txt', 'fra-x-bible-pirotclamer-v1.txt', 'gub-GUBWBT.txt', 'guj-x-bible-guj-v1.txt', 'gur-x-bible-frafra-v1.txt', 'hat-x-bible-1985-v1.txt', 'hat-x-bible-1999-v1.txt', 'hmo-x-bible-hmo-v1.txt', 'hrv-x-bible-hrv-v1.txt', 'hui-HUIPNG.txt', 'hun-x-bible-2005-v1.txt', 'hun-x-bible-karoli-v1.txt', 'ifa-IFAWBT.txt', 'ifb-IFBTBL.txt', 'ify-IFYWBT.txt', 'ify-x-bible-ify-v1.txt', 'ind-x-bible-suciinjil-v1.txt', 'ind-x-bible-terjemahanbaru-v1.txt', 'ita-x-bible-2009-v1.txt', 'ita-x-bible-diodati-v1.txt', 'ita-x-bible-nuovadiodati1991-v1.txt', 'ita-x-bible-riveduta-v1.txt', 'kac-KACUBS.txt', 'kaz-x-bible-kaz-v1.txt', 'kek-x-bible-1988-v1.txt', 'kek-x-bible-2005-v1.txt', 'kjb-x-bible-kjb-v1.txt', 'kor-x-bible-revised-v1.txt', 'lat-x-bible-novavulgata-v1.txt', 'lat-x-bible-vulgataclementina-v1.txt', 'lit-x-bible-lit-v1.txt', 'lnd-LNDBSM.txt', 'lsi-LSIBSM.txt', 'mad-MADIBS.txt', 'mah-x-bible-mah-v1.txt', 'mam-x-bible-northern-v1.txt', 'may-ZLMAVB.txt', 'mdy-MDYBSE.txt', 'mlg-MLGRCV.txt', 'mlg-MLGRPV.txt', 'mps-x-bible-mps-v1.txt', 'mri-x-bible-mri-v1.txt', 'mrw-MRWNVS.txt', 'mya-x-bible-mya-v1.txt', 'nhe-NHETBL.txt', 'nld-x-bible-nld-v1.txt', 'nor-x-bible-nor-v1.txt', 'nor-x-bible-student-v1.txt', 'pis-x-bible-pis-v1.txt', 'plt-x-bible-romancatholic-v1.txt', 'poh-POHPOC.txt', 'poh-x-bible-eastern-v1.txt', 'por-x-bible-almeidaatualizada-v1.txt', 'por-x-bible-almeidarevista-v1.txt', 'por-x-bible-paratodos-v1.txt', 'prs-PRSGNN.txt', 'pxm-PXMBSM.txt', 'qub-x-bible-qub-v1.txt', 'quh-QUHSBB.txt', 'quh-x-bible-1993-v1.txt', 'quy-x-bible-quy-v1.txt', 'quz-QUZPBS.txt', 'quz-x-bible-quz-v1.txt', 'qxr-QXRBSE.txt', 'ron-x-bible-cornilescu-v1.txt', 'rug-RUGWBT.txt', 'rus-RUSS76.txt', 'rus-x-bible-synodal-v1.txt', 'som-SOMSIM.txt', 'som-x-bible-som-v1.txt', 'suz-SUZWBT.txt', 'swe-SWESFB.txt', 'swe-SWESFV.txt', 'swe-x-bible-folk1998-v1.txt', 'tat-TTRIBT.txt', 'tbz-TBZBSB.txt', 'tbz-x-bible-tbz-v1.txt', 'tcw-x-bible-tcw-v1.txt', 'tgl-x-bible-1905-v1.txt', 'tlh-x-bible-klingon-v1.txt', 'tpi-x-bible-tpi-v1.txt', 'tpm-TPMWBT.txt', 'tpm-x-bible-tpm-v1.txt', 'tur-x-bible-southernazeri-v1.txt', 'tzo-TZESBM.txt', 'ukr-x-bible-1962-v1.txt', 'ukr-x-bible-2009-v1.txt', 'vie-x-bible-1926compounds-v1.txt', 'vie-x-bible-1926nocompounds-v1.txt', 'vie-x-bible-2002-v1.txt', 'wal-x-bible-wal-v1.txt', 'wbm-x-bible-wbm-v1.txt', 'xho-x-bible-1996-v1.txt', 'xho-x-bible-xho-v1.txt', 'yua-YUASBM.txt', 'zom-x-bible-zom-v1.txt']
languages=[i[:-4] for i in languages] #remove .txt

# Create embeddings for each language
for language in languages:
	print(language)

	# Create embeddings for each seed for that language
	for seed in seeds:
		print(seed)
		
		# Read in Bible text
		with open(nicolai_path+language+'.txt','r') as text_file:
			sentences = text_file.readlines()
			sentences = [i[:-1].split(' ') for i in sentences]

		# Create word2vec embedding space
		model = word2vec.Word2Vec(sentences,size=300,window=5,min_count=5,seed=seed)

		# Save model for future use
		with open(output_path+language+'_'+str(seed)+'.pkl', 'wb') as pickle_file:
			pickle.dump(model,pickle_file)
