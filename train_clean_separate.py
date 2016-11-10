# -*- coding: UTF-8 -*- 
#
# Script to clean, separate train data by months, write as csv
#

# Python & Co
import os
import numpy as np
import pandas as pd
import logging

# Project
from data_clean import clean


logging.basicConfig(level=logging.INFO)

# **Dataset Size:**
# 
# First let us check the number of rows in train and test file :
# - Number of rows in train :  13'647309
# - Number of rows in test :  929615
# - Number of clients (train dataset) : 956645

# **Dataset columns:**
#   
#   
# Main columns :   
# 
# - fecha_dato 	The table is partitioned for this column
# - ncodpers 	Customer code
# - ind_empleado 	Employee index: A active, B ex employed, F filial, N not employee, P pasive
# - pais_residencia 	Customer's Country residence
# - sexo 	Customer's sex
# - age 	Age
# - fecha_alta 	The date in which the customer became as the first holder of a contract in the bank
# - ind_nuevo 	New customer Index. 1 if the customer registered in the last 6 months.
# - antiguedad 	Customer seniority (in months)
# - indrel 	1 (First/Primary), 99 (Primary customer during the month but not at the end of the month)
# - ult_fec_cli_1t 	Last date as primary customer (if he isn't at the end of the month)
# - indrel_1mes 	Customer type at the beginning of the month ,1 (First/Primary customer), 2 (co-owner ),P (Potential),3 (former primary), 4(former co-owner)
# - tiprel_1mes 	Customer relation type at the beginning of the month, A (active), I (inactive), P (former customer),R (Potential)
# - indresi 	Residence index (S (Yes) or N (No) if the residence country is the same than the bank country)
# - indext 	Foreigner index (S (Yes) or N (No) if the customer's birth country is different than the bank country)
# - conyuemp 	Spouse index. 1 if the customer is spouse of an employee
# - canal_entrada 	channel used by the customer to join
# - indfall 	Deceased index. N/S
# - tipodom 	Addres type. 1, primary address
# - cod_prov 	Province code (customer's address)
# - nomprov 	Province name
# - ind_actividad_cliente 	Activity index (1, active customer; 0, inactive customer)
# - renta 	Gross income of the household
# - segmento 	segmentation: 01 - VIP, 02 - Individuals 03 - college graduated    
#     
#     
# target columns : 
#     
# - ind_ahor_fin_ult1 	Saving Account
# - ind_aval_fin_ult1 	Guarantees
# - ind_cco_fin_ult1 	Current Accounts
# - ind_cder_fin_ult1 	Derivada Account
# - ind_cno_fin_ult1 	Payroll Account
# - ind_ctju_fin_ult1 	Junior Account
# - ind_ctma_fin_ult1 	Más particular Account
# - ind_ctop_fin_ult1 	particular Account
# - ind_ctpp_fin_ult1 	particular Plus Account
# - ind_deco_fin_ult1 	Short-term deposits
# - ind_deme_fin_ult1 	Medium-term deposits
# - ind_dela_fin_ult1 	Long-term deposits
# - ind_ecue_fin_ult1 	e-account
# - ind_fond_fin_ult1 	Funds
# - ind_hip_fin_ult1 	Mortgage
# - ind_plan_fin_ult1 	Pensions
# - ind_pres_fin_ult1 	Loans
# - ind_reca_fin_ult1 	Taxes
# - ind_tjcr_fin_ult1 	Credit Card
# - ind_valo_fin_ult1 	Securities
# - ind_viv_fin_ult1 	Home Account
# - ind_nomina_ult1 	Payroll
# - ind_nom_pens_ult1 	Pensions
# - ind_recibo_ult1 	Direct Debit    
#     

TARGET_LABELS = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 
                 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
                 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
                 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
                 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
                 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
                 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']


## Data stats: data per month representations
logging.info("Setup data per month representations")

data_path = "data/"
train = pd.read_csv(data_path+"train_ver2.csv", usecols=['fecha_dato', 'ncodpers'], parse_dates=['fecha_dato'])

# Setup month start/end row indices
gb = train.groupby('fecha_dato')
data_count_per_month = gb.agg('size')

month_start_end_row_indices = {}
for key, group_value in gb:
    month_start_end_row_indices[key] = [group_value.index[0], group_value.index[-1]]

del train
del gb


## Clean and save data per month
load_dtypes={"sexo":str, "ind_nuevo":str, "ult_fec_cli_1t":str, "indext":str, "indrel_1mes":str, "conyuemp":str}

for month_key in month_start_end_row_indices:
    
    logging.info("Process the month: %s" % month_key)
    path = "train_%s-%s.csv" % (month_key.year, month_key.month)
    filename=data_path+path
    if os.path.exists(filename):
        logging.info("-- Found existing file: %s" % filename)
        continue
    
    # Read a month
    logging.info("- Read data")
    skiprows = month_start_end_row_indices[month_key][0]
    nrows = month_start_end_row_indices[month_key][1] - skiprows + 1
    train_month = pd.read_csv(data_path+"train_ver2.csv", dtype=load_dtypes, skiprows=range(1, skiprows+1), nrows=nrows)

    train_month["fecha_dato"] = pd.to_datetime(train_month["fecha_dato"],format="%Y-%m-%d")
    train_month["fecha_alta"] = pd.to_datetime(train_month["fecha_alta"],format="%Y-%m-%d")
    train_month["age"] = pd.to_numeric(train_month["age"], errors="coerce")

    # Data Cleaning
    df = train_month
    clean(df)
    assert df.isnull().any().sum() == 0, "Data still contains nan values : \n\n {}".format(df.isnull().any())
    
    logging.info("- Write data")
    df.to_csv(filename, index_label=False)   
    




