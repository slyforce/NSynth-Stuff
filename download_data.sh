#!/bin/bash 

mkdir -p data; 

trg_folder=$(realpath data);
for f in test valid train; do 
	echo "Downloading $f to ${trg_folder}";

	fname="${trg_folder}/nsynth-$f.jsonwav.tar.gz" 
	wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-$f.jsonwav.tar.gz -O $fname;

	echo "Unzipping data"
	tar xfvz $fname; 
done 