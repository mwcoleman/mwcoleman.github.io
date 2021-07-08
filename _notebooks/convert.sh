#!/bin/sh

ARGS=("$@") # Capture all args passed in, for YAML config

source "/home/matt/venv/tf/bin/activate"

nb=${ARGS[0]} 

function convert(){
	jupyter nbconvert --to markdown $nb.ipynb
	
	python edit.py ${nb%.ipynb}.md ${ARGS[@]:1}  # Slice to the end of the array https://stackoverflow.com/questions/1335815/how-to-slice-an-array-in-bash
#	mv ${nb%.ipynb}.md
	mv ${nb%.ipynb}_files ../images/
	echo "==========Conversion complete!=========="
}

convert
