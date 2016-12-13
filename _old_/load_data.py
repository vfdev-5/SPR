#
# Scripts to load data
# a) load_all() : loads all 'reduced_train_201X-XX.csv' files
# b) load_month(month, reduced=False) : loads a month (0-16) data 'train_201X-XX.csv' or 'reduced_train_201X-XX.csv'


TARGET_LABELS = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

import os
import numpy as np
import pandas as pd

DATA_PATH = 'data'


def load_all():
    """
    Method to load all 'reduced_train_201X-XX.csv' files 
    """
    files = os.listdir(DATA_PATH)
    out_df = pd.DataFrame()
    for f in files:
        if "reduced_train_" in f:
            df = pd.read_csv(os.path.join(DATA_PATH, f), parse_dates=['fecha_dato', 'fecha_alta'])
            out_df = pd.concat([out_df, df], axis=0)
            
    return out_df        
            

def load_month(month, reduced=False):
    """
    Method to load a month (0-16) data 'train_201X-XX.csv' or 'reduced_train_201X-XX.csv'
    """
    filename = 'train_2015-%s.csv' % (month+1) if month < 12 else 'train_2016-%s.csv' % (month-11)
    if reduced:
        filename = 'reduced_' + filename
    return pd.read_csv(os.path.join(DATA_PATH, filename), parse_dates=['fecha_dato', 'fecha_alta'])
    
    
def load_all_encoded():
    """
    Method to load 'encoded_reduced_train_all.csv' file
    """
    return pd.read_csv(os.path.join(DATA_PATH, "encoded_reduced_train_all.csv"), parse_dates=['fecha_dato', 'fecha_alta'])    