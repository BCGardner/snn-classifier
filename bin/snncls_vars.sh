#!/bin/bash
# source this script in .bashrc

# Change this absolute path to where project was cloned locally
export SNNCLS_DIR=/home/brian/Documents/git_repos/snn-classifier

# Python modules
export PYTHONPATH=$SNNCLS_DIR/lib:$PYTHONPATH

# Make executables visible
export PATH=$SNNCLS_DIR/bin:$PATH
