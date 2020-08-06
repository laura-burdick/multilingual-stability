#!/bin/bash
# Laura Burdick (lburdick@umich.edu)
# Create five GloVe embedding spaces (downsampling without replacement) for each language in Wikipedia.

# SET THESE VARIABLES

# Location of the Wikipedia data
# Can be downloaded from https://sites.google.com/site/rmyeid/projects/polyglot
# Before running this code, for each language, you need to create five downsamples (without replacement).
# These should be stored with the file names {nicolai_path}{language}/downsampled_without_replacement_{downsample_index}.txt
# (This script uses the downsample indices 0-4, though this can be changed below)
nicolai_path=/local/data/polyglot/

# Location where output embedding spaces will be stored
# For each embedding space, multiple files will be created:
# A vocab file, stored as {output_path}{language}/downsampled_without_replacement_glove_vocab_{downsample_index}.txt
# A coocurrence file, stored as {output_path}{language}/downsampled_without_replacement_glove_coocurrence_{downsample_index}.bin
# A shuffled coocurrence file, stored as {output_path}{language}/downsampled_without_replacement_glove_coocurrence_{downsample_index}.shuf.bin
# The final embedding space, stored as {output_path}{language}/downsampled_without_replacement_glove_{downsample_index}
output_path=nicolai_path

# Location of code for GloVe
# Can be downloaded from https://nlp.stanford.edu/projects/glove/
glove_path=/home/wenlaura/embedding-spaces/NAACL2018/GloVe

# Languages to create embedding spaces for (can adjust if needed, or leave the same)
languages=(ar bg ca cs da de el en es et fa fi fr he hi hr hu id it ja ko lt lv ms nl no pl pt ro ru sk sl sr sv th tl tr uk vi zh)

# Indices of downsamples to process (can adjust if needed, or leave the same)
# Right now, we are processing five downsamples
indices=(0 1 2 3 4)

# Create embeddings for each language
for language in ${languages[@]};
do
	echo ${language}

	# Create embeddings for each downsample
	for index in ${indices[@]};
	do
		${glove_path}/build/vocab_count -min-count 5 -verbose 2 < ${nicolai_path}${language}/downsampled_without_replacement_${index}.txt > ${output_path}${language}/downsampled_without_replacement_glove_vocab_${index}.txt
		${glove_path}/build/cooccur -verbose 2 -window-size 5 -vocab-file ${output_path}${language}/downsampled_without_replacement_glove_vocab_${index}.txt < ${nicolai_path}${language}/downsampled_without_replacement_${index}.txt > ${output_path}${language}/downsampled_without_replacement_glove_coocurrence_${index}.bin
		${glove_path}/build/shuffle -verbose 2 < ${output_path}${language}/downsampled_without_replacement_glove_coocurrence_${index}.bin > ${output_path}${language}/downsampled_without_replacement_glove_coocurrence_${index}.shuf.bin
		${glove_path}/build/glove -verbose 2 -vector-size 300 -iter 100 -seed 2518 -vocab-file ${output_path}${language}/downsampled_without_replacement_glove_vocab_${index}.txt -input-file ${output_path}${language}/downsampled_without_replacement_glove_coocurrence_${index}.shuf.bin -save-file ${output_path}${language}/downsampled_without_replacement_glove_${index}
	done
done
