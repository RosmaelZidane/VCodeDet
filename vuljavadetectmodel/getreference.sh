#!/bin/bash

# The Google Drive file ID
FILE_ID="1ypL7B_eYwFEz7yI7AA7OlGN9jli-zdc6"   # proccessed folder 
FILE_ID2="1lW3TzeEbSqDv88obN6nL5M3FMI-EMZ5c"  # cache folder   
# FILE_ID3="1I5IYfDJrvCrhL8fDPivr71ojD_o0ygnb"  # model checkpoit
FILE_ID3="1dIGRkg_YqBcUWThnwqRPyQD6_rhIqGfI"    # model checkpoit



# The name of the output zip file
OUTPUT_FILE="shared_folder.zip"
OUTPUT_FILE2="shared_folderr.zip"
OUTPUT_FILE3="vjavadet.ckpt"


cd vuljavadetectmodel/storage

if [[ ! -d processed ]]; then
    echo "Downloading the file from Google Drive..."
	gdown --id $FILE_ID -O $OUTPUT_FILE
	unzip $OUTPUT_FILE 
    rm shared_folder.zip
else
    echo "Already downloaded processed"
fi

if [[ ! -d cache ]]; then
    echo "Downloading the file from Google Drive..."
	gdown --id $FILE_ID2 -O $OUTPUT_FILE2
	unzip $OUTPUT_FILE2 
    rm shared_folderr.zip
else
    echo "Already downloaded processed"
fi


cd external
mkdir checkpoints
cd checkpoints

if [[ ! -d checkpoints ]]; then
    echo "Downloading the file from Google Drive..."
    gdown --id $FILE_ID3 -O $OUTPUT_FILE3
    rm -rf OUTPUT_FILE3
else
    echo "Already downloaded model checkpoint"
fi




# Check if the download was successful
if [ $? -ne 0 ]; then
  echo "Download failed. Exiting..."
  exit 1
fi

echo "Download complete!"
