# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 2024

@author: Jonas A. Thiele
"""

from sklearn.model_selection import StratifiedKFold
import pandas as pd

#Create stratified folds
def create_folds(df, n_s, n_grp):
    
    df['Fold'] = -1
    
    skf = StratifiedKFold(n_splits=n_s, shuffle=True, random_state=None)
    df['grp'] = pd.cut(df.target, n_grp, labels=False)
    target = df.grp
    
    for fold_no, (t, v) in enumerate(skf.split(target, target)):
        df.loc[v, 'Fold'] = fold_no
    return df