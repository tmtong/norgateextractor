#!/bin/bash

USER=`whoami`
# Define paths
VENV_PATH="/home/${USER}/Documents/norgateextractor/.venv"
SCRIPT_PATH="/home/${USER}/Documents/norgateextractor/norgateextractor/dailyindex.py"

# Activate the virtual environment
# source "$VENV_PATH/bin/activate" || { echo "Failed to activate virtual environment"; exit 1; }


echo "Starting Python application at $(date)"
python "$SCRIPT_PATH"
