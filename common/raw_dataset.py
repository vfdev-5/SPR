
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


def parse_line(line):
    """
    :param line:
    :return: array of values
    """
    if "\"" in line:
        tmp1 = line.split("\"")
        arr = tmp1[0][:-1].split(",") + [tmp1[1]] + tmp1[2][1:].split(',')
    else:
        arr = line.split(",")
    arr = [a.strip() for a in arr]
    return arr


def get_target_labels(header_line):
    """
    :param header_line:
    :return:
    """
    line = header_line.strip()
    line = line.replace("\"", "")
    return line.split(",")[24:]


def get_user(row):
    """
    :param row:
    :return:
    """
    return row[1]


def get_profile(row):
    return row[2:24]


def get_choices(row):
    return row[24:]
