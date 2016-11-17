# -*- coding: UTF-8 -*- 
#
#
#
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


def clean(df):
    # There are some 'unknown' users in train dataset only
    unknown_users = df['sexo'].isnull() & df['age'].isnull() & df['ind_empleado'].isnull() & df['fecha_alta'].isnull() & df['pais_residencia'].isnull()

    logging.info("- Number of unknown clients : %s" % unknown_users.sum())

    # **Remove these users** !
    df.drop(df[unknown_users].index, inplace=True)

    logging.info("- Number of columns with nan : %s" % df.isnull().any().sum())

    # Remove accent
    df.loc[df['nomprov']=="CORU\xc3\x91A, A","nomprov"] = "CORUNA"
    
    unknown_cols = ['sexo', 
                    'ind_empleado', 
                    'pais_residencia', 
                    'ult_fec_cli_1t', 
                    'conyuemp', 
                    'canal_entrada', 
                    'nomprov', 
                    'segmento',
                    'tiprel_1mes',
                    'indrel_1mes']
    # Start with cols -> replace nan with UNKNOWN
    for col in unknown_cols:
        df.loc[df[col].isnull(),col] = "UNKNOWN"

    # Continue with age -> replace nan with mean , less 18 -> mean between 18, 30 and greater than 90 -> mean between 30, 90
    df.loc[df.age < 18, "age"]  = df.loc[(df.age >= 18) & (df.age <= 30),"age"].mean(skipna=True)
    df.loc[df.age > 90, "age"] = df.loc[(df.age >= 30) & (df.age <= 90),"age"].mean(skipna=True)
    df["age"].fillna(df["age"].mean(),inplace=True)
    df["age"] = df["age"].astype(int)
    
    # Next `fecha_alta` : 
    assert df['fecha_alta'].isnull().sum() == 0, "Need to replace nan in 'fecha_alta', count=%s" % df['fecha_alta'].isnull().sum()
    
    # **Remove 'tipodom' and 'cod_prov' columns**
    df.drop(["tipodom","cod_prov"],axis=1,inplace=True)

    # **Remove clients not staying in Spain with known (spanish) `nomprov`**
    mask = (df['nomprov'] != "UNKNOWN") & (df['pais_residencia'] != "ES")
    logging.info("- Remove clients not staying in Spain with known (spanish) nomprov : count = %s" % mask.sum())
    df.drop(df[mask].index, inplace=True)
    
    # Fix problems with 'indrel_1mes' and 'tiprel_1mes' :
    df['fecha_alta'] = pd.to_datetime(df['fecha_alta'])
    df['fecha_dato'] = pd.to_datetime(df['fecha_dato'])
    mask = (df['indrel_1mes'] == "UNKNOWN") & (df['tiprel_1mes'] == "UNKNOWN") & (df['fecha_dato'].dt.month == df['fecha_alta'].dt.month)
    df.loc[mask & (df['indrel'] == 1), 'indrel_1mes'] = '1'
    df.loc[mask & (df['indrel'] == 99), 'indrel_1mes'] = '3'
    df.loc[mask, 'tiprel_1mes'] = 'A'
    
    
    # Target labels : `ind_nomina_ult1`, `ind_nom_pens_ult1` : nan -> 0
    # I could try to fill in missing values for products by looking at previous months, but since it's such a small number of values for now I'll take the cheap way out.
    df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
    df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0


    # ### Now for gross income, aka `renta`
    # Fill `renta` nan -> median per region, employee index, segment, gender, if has no information -> replace by -99
    incomes_gb = df[df['renta'].notnull()].groupby(['nomprov', 'ind_empleado', 'segmento', 'sexo', 'age'])
    incomes_stats = incomes_gb.agg("median")
    nan_incomes_gb = df[df['renta'].isnull()].groupby(['nomprov', 'ind_empleado', 'segmento', 'sexo', 'age'])
    # nan_incomes_stats = nan_incomes_gb.agg("size")

    for key, group_value in nan_incomes_gb:
        if key in incomes_stats.index:
            df.loc[group_value.index, 'renta'] = incomes_stats.loc[key]['renta']
        else:
            df.loc[group_value.index, 'renta'] = -99

    df['logrenta'] = np.log(df[df['renta'] > 0]['renta'] + 1)
    df.loc[df['logrenta'].isnull(), 'logrenta'] = -99
    # Drop 'renta' and 'antiguedad'
    df.drop(['renta', 'antiguedad'], axis=1, inplace=True)        
                
        