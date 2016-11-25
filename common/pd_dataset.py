
#
#
#

# Python
import logging
from sklearn.preprocessing import LabelEncoder
# Project
from utils import to_yearmonth, to_ym_dec, to_nb_months

TARGET_LABELS = [
    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
    'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
    'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
    'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
    'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
    'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
    'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'
]


def minimal_clean_data_inplace(df):
    """
    Script to clean data in input DataFrame
    """
    # There are some 'unknown' users in train dataset only
    unknown_data_lines = df['sexo'].isnull() & df['age'].isnull() & df['ind_empleado'].isnull() & \
                    df['fecha_alta'].isnull() & df['pais_residencia'].isnull()

    logging.info("- Number of lines with unknown data : %s" % unknown_data_lines.sum())

    # Remove these users as clients
    _clients = df[unknown_data_lines]['ncodpers'].unique()
    bad_lines = df['ncodpers'].isin(_clients)
    df.drop(df[bad_lines].index, inplace=True)
    # df.drop(df[unknown_users].index, inplace=True)

    logging.info("- Number of columns with nan : %s" % df.isnull().any().sum())

    # Remove accent
    df.loc[df['nomprov'] == "CORU\xc3\x91A, A", "nomprov"] = "CORUNA"

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
        df.loc[df[col].isnull(), col] = "UNKNOWN"

    # Set unknown renta to -99
    df.loc[df['renta'].isnull(), 'renta'] = -99

    # Next `fecha_alta` :
    assert df['fecha_alta'].isnull().sum() == 0, \
        "Need to replace nan in 'fecha_alta', count=%s" % df['fecha_alta'].isnull().sum()

    # **Remove 'tipodom' and 'cod_prov' columns**
    df.drop(["tipodom", "cod_prov"], axis=1, inplace=True)

    if "ind_nomina_ult1" in df.columns and "ind_nom_pens_ult1" in df.columns:
        # Target labels : `ind_nomina_ult1`, `ind_nom_pens_ult1` : nan -> 0
        # I could try to fill in missing values for products by looking at previous months,
        # but since it's such a small number of values for now I'll take the cheap way out.
        df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
        df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0

    # replace 'antiguedad' with the number of months between 'fecha_alta' and 'fecha_dato'
    def _compute_duration(row):
        ym1 = to_yearmonth(row['fecha_alta'])
        ym2 = to_yearmonth(row['fecha_dato'])
        ym_dec1 = to_ym_dec(ym1)
        ym_dec2 = to_ym_dec(ym2)
        return to_nb_months(ym_dec2 - ym_dec1)
    df['antiguedad'] = df.apply(_compute_duration, axis=1)


def preprocess_data_inplace(df):
    """
    Script to process data in input DataFrame
    """
    string_data = df.drop(['fecha_dato', 'fecha_alta'], axis=1).select_dtypes(include=["object"])
    for c in string_data.columns:
        le = LabelEncoder()
        le.fit(df[c])
        df[c] = le.transform(df[c])
