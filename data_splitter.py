import random as rnd
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import os

# Source directory
source_dir = "fish_xml"

# Train directory
train_dir = "train"
if not os.path.isdir(train_dir):
    os.mkdir(train_dir) 

# Test directory
test_dir = "test"
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# Get all files and their paths from a directory
def get_files_in_dir(dir):
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    paths = [join(dir, f) for f in files]
    return files, paths

# Careful: Remove all files from a directory (used to clean up train and test dir)
def clear_directory(dir):
    files, paths = get_files_in_dir(dir)

    for p in paths:
        os.remove(p)

# Clean up train and test dirs
clear_directory("train")
clear_directory("test")

# Iterate over all files in the source directory and copy them to either test or train dir
files, paths = get_files_in_dir(source_dir)
for file in files:
    src_file = join(source_dir, file)
    
    # Copy about 20% to test dir
    if (rnd.random() >= 0.8):
        copyfile(src_file, join(test_dir, file))
    # Copy about 80% to train dir
    else:
        copyfile(src_file, join(train_dir, file))
