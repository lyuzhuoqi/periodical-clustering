from gensim.models import KeyedVectors
from utils import prinT, get_data_dir
import os
import pandas as pd
import pickle

def WordVector(decade: str, d: int, w: int):
    '''
    Load pre-trained wordvectors of periodicals.
    start_year, end_year: e.g. 1950, 1959
    d: number of dimension
    w: window size
    '''
    prinT("start loading word vectors...")
    wv = KeyedVectors.load(os.path.join(get_data_dir(), decade, '%dfeat_%dcontext_win_size')%(d, w))
    prinT('word vectors loaded, and its shape is: ' + str(wv.vectors.shape))
    return wv

def MAG_venue_df():
    '''
    Load MAG venue dataframe.
    '''
    with open(os.path.join(get_data_dir(), "MAG_venue_info_df.pkl"), "rb") as f:
        MAG_venue_df = pickle.load(f)
    return MAG_venue_df