#!/bin/bash

# exit on errors, unreferenced variables, and pipe errors
set -euo pipefail

data_folder="/mnt/c/work/code/X-maps/ESL_data"
dynamic_folder="$data_folder/dynamic"
mkdir -p $dynamic_folder

esl_data_url="https://rpg.ifi.uzh.ch/data/esl/dynamic"

echo "Downloading and extracting data to ${dynamic_folder} ..."

for seq_names in "seq1 fan"
# for seq_names in "seq1 book_duck"
do
    tuple=( $seq_names );
    raw_url="${esl_data_url}/${tuple[1]}/data.raw"
    bias_url="${esl_data_url}/${tuple[1]}/data.bias"
    dest_folder="${dynamic_folder}/${tuple[0]}/"
    wget --no-clobber "$raw_url" -P "$dest_folder"
    wget --no-clobber "$bias_url" -P "$dest_folder"
done
