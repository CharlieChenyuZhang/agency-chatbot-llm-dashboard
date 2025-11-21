#!/bin/bash

# Script to count files in the specified directory
# DIR="data/dataset/final_trainingdata_gpt5.1_1000_per_category_nov20/gpt5_goal_persistence_1"
DIR="data/dataset/final_trainingdata_gpt5.1_1000_per_category_nov20/gpt5_independence_1"
# DIR="data/dataset/final_trainingdata_gpt5.1_1000_per_category_nov20/gpt5_rigidity_1"

# Count files (excluding directories)
FILE_COUNT=$(find "$DIR" -maxdepth 1 -type f | wc -l)

echo "Number of files in $DIR: $FILE_COUNT"

