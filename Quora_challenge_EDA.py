'''
Problem Statement
- Identify which questions asked on Quora are duplicates of questions that have already been asked.
- This could be useful to instantly provide answers to questions that have already been answered.
- We are tasked with predicting whether a pair of questions are duplicates or not.

Dataset : https://www.kaggle.com/c/quora-question-pairs

Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate
- Size of Train.csv - 60MB
- Number of rows in Train.csv = 404,290



'''



import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
import gc

import sys
sys.path.append(r'C:\Users\user\Google Drive\my projects\data_wrangling_tools')
import file_sys as fsys

import re

DATA_DIR = Path(r'C:\Users\user\Documents\Data\data_Quora_question_pairs_challenge_kaggle')
GLOVE_DIR = DATA_DIR.parent/'glove_wiki_6B'
GLOVE_fpath = GLOVE_DIR/"glove.6B.100d.txt"
GLOVE2word2vec_vec_path = GLOVE_DIR/"gensim_glove2word2vec_vectors.txt"
wv_path = Path()/'word_vectors.pkl'
wmdistance_path = Path()/'wmdistance_q_pairs.pkl'
NSAMPLES = 10


ftrain = 'train.csv'
ftest = 'test.csv'

df = pd.read_csv(DATA_DIR/ftrain).sample(NSAMPLES, random_state=23)

import gensim
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

if not GLOVE2word2vec_vec_path.exists():
    glove2word2vec(glove_input_file=GLOVE_fpath, word2vec_output_file=GLOVE2word2vec_vec_path)

if wv_path.exists():
    word_vectors = fsys.load_pickle(path=wv_path)

else:
    # Load a word2vec model stored in the C *text* format.
    word_vectors = KeyedVectors.load_word2vec_format(GLOVE_DIR/'gensim_glove2word2vec_vectors.txt', binary=False)
    fsys.save_pickle(path=wv_path, var=word_vectors)

# add a column with word-mover-distance (WMD):
q_pair = list(zip(df['question1'].to_list(), df['question2'].to_list())) #make a list with tuples of q1,q2
#iterate over q1,q2 tuples and calculate wmdistance:
if wmdistance_path.exists():
    wmd_dist = fsys.load_pickle(path=wmdistance_path)
else:
    wmd_dist = [word_vectors.wmdistance(*q) for q in q_pair]
    fsys.save_pickle(path=wmdistance_path, var=wmd_dist)

df['wmd_dist'] = wmd_dist

import plotly.express as px
fig = px.histogram(df, x="wmd_dist", color="is_duplicate",
                   marginal="box", # or violin, rug
                   hover_data=df.columns)
fig.show()

print('done')
