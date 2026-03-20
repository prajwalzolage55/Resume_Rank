#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# Download Spacy model
python -m spacy download en_core_web_sm

# Download NLTK data (optional if handle differently in app.py, but safer here)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
