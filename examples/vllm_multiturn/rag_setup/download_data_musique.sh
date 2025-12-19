#!/bin/bash
# This script is taken from https://github.com/StonyBrookNLP/musique with slight modifications

set -e
set -x

# If gdown doesn't work, you can download files from mentioned URLs manually
# and put them at appropriate locations.
pip install gdown

ZIP_NAME="musique_v1.0.zip"

# URL: https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing
gdown --id 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h --output $ZIP_NAME

TARGET_DIR="./data/raw"
mkdir -p $TARGET_DIR
unzip -o $(basename $ZIP_NAME) -d $TARGET_DIR # Extract directly into target

# Move contents from the extracted 'data' folder up one level
mv $TARGET_DIR/data/* $TARGET_DIR/

# Clean up the empty directory and the zip
rm -rf $TARGET_DIR/data
rm $ZIP_NAME

# TODO: prevent these from zipping in.
rm -rf __MACOSX
# Clean up potential extracted .DS_Store
rm -f $TARGET_DIR/.DS_Store
