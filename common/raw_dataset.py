
import math

from utils import to_yearmonth


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


def get_yearmonth(row):
    return to_yearmonth(row[0])


def get_profile(row):
    return row[2:24]


def get_choices(row):
    return row[24:]


def clean_data(input_row):
    row = list(input_row)
    clean_data_inplace(row)
    return row


def clean_data_inplace(row):
    """
    Method to clean data rows
    """
    ll = len(row)
    row_indices = range(ll)
    # Replace empty values by NA
    for i in row_indices:
        if row[i] == '':
            row[i] = "NA"

    (fecha_dato, ncodpers, ind_empleado,  # 0
     pais_residencia, sexo, age,  # 3
     fecha_alta, ind_nuevo, antiguedad,  # 6
     indrel, ult_fec_cli_1t, indrel_1mes,  # 9
     tiprel_1mes, indresi, indext,  # 12
     conyuemp, canal_entrada, indfall,  # 15
     tipodom, cod_prov, nomprov,  # 18
     ind_actividad_cliente, renta, segmento) = row[:24]

    # If unknown user <=> no info in columns ['age', 'ind_empleado', 'fecha_alta', 'pais_residencia', 'sexo']
    if len(fecha_alta) + len(age) + len(ind_empleado) + len(pais_residencia) + len(sexo) == len('NA')*5:
        # remove data
        return False

    # Remove clients not staying in Spain with known (spanish) nomprov
    if nomprov != "NA" and pais_residencia != "ES":
        # remove data
        return False

    # Convert to types :
    row[1] = int(ncodpers)
    row[22] = float(renta) if renta != "NA" else renta
    # Remove floating point at string indrel_1mes
    row[11] = str(int(float(indrel_1mes))) if len(indrel_1mes) == 3 else indrel_1mes

    for i in range(24, ll):
        row[i] = int(row[i]) if row[i] != "NA" else 0

    (fecha_dato, ncodpers, ind_empleado,  # 0
     pais_residencia, sexo, age,  # 3
     fecha_alta, ind_nuevo, antiguedad,  # 6
     indrel, ult_fec_cli_1t, indrel_1mes,  # 9
     tiprel_1mes, indresi, indext,  # 12
     conyuemp, canal_entrada, indfall,  # 15
     tipodom, cod_prov, nomprov,  # 18
     ind_actividad_cliente, renta, segmento) = row[:24]

    # Fix accents for 'nomprov'
    if nomprov == "CORU\xc3\x91A, A":
        row[20] = "CORUNA"

    # Clamp age
    # if age < 18:
    #     row[5] = MEAN_AGE_18_30
    # elif age > 90:
    #     row[5] = MEAN_AGE_31_90

    # Fix problems with 'indrel_1mes' and 'tiprel_1mes':
    fecha_alta_month = fecha_alta[5:7]
    fecha_dato_month = fecha_dato[5:7]

    if indrel_1mes == "NA" and tiprel_1mes == "NA" and fecha_alta_month == fecha_dato_month:
        if indrel == '1':
            row[11] = '1'  # indrel_1mes = '1'
        elif indrel == '99':
            row[11] = '3'  # indrel_1mes = '3'
        row[12] = 'A'  # tiprel_1mes = 'A'

    (fecha_dato, ncodpers, ind_empleado,  # 0
     pais_residencia, sexo, age,  # 3
     fecha_alta, ind_nuevo, antiguedad,  # 6
     indrel, ult_fec_cli_1t, indrel_1mes,  # 9
     tiprel_1mes, indresi, indext,  # 12
     conyuemp, canal_entrada, indfall,  # 15
     tipodom, cod_prov, nomprov,  # 18
     ind_actividad_cliente, renta, segmento) = row[:24]

    # Fill `renta` nan -> median per region, employee index, segment, gender, if has no information -> replace by -99
    if renta == "NA":
        # age_group = get_age_group_index(age)
        # key = (pais_residencia, nomprov, ind_empleado, segmento, sexo, age_group)
        # if key in INCOMES_STATS_MAP:
        #     row[22] = INCOMES_STATS_MAP[key]
        # else:
        #     row[22] = -99.0
        row[22] = -99.0
    return True


def process_data(input_row):
    row = list(input_row)
    process_data_inplace(row)
    return row


def process_data_inplace(row):
    """
    Method to process data rows (feature engineering)

    (fecha_dato, ncodpers, ind_empleado,  # 0
     pais_residencia, sexo, age,  # 3
     fecha_alta, ind_nuevo, antiguedad,  # 6
     indrel, ult_fec_cli_1t, indrel_1mes,  # 9
     tiprel_1mes, indresi, indext,  # 12
     conyuemp, canal_entrada, indfall,  # 15
     tipodom, cod_prov, nomprov,  # 18
     ind_actividad_cliente, renta, segmento)  # 21

    renta -> income group index
    age -> age group index
    antiguedad -> int(fecha_dato - fecha_alta)

    """
    row[22] = get_income_group_index(row[22])
    row[5] = get_age_group_index(row[5])
    res = to_yearmonth(row[0])*0.01 - to_yearmonth(row[6])*0.01
    row[8] = int(math.floor(res)) * 12 + int(math.ceil((res - int(res)) * 100))


def get_age_group_index(age):
    """
    0 : 18 - 23
    1 : 23 - 28
    2 : 28 - 32
    3 : 32 - 40
    4 : 40 - 50
    5 : 50 - 60
    6 : 60 - ...
    """
    if age < 23:
        return 0
    elif age < 28:
        return 1
    elif age < 32:
        return 2
    elif age < 40:
        return 3
    elif age < 50:
        return 4
    elif age < 60:
        return 5
    else:
        return 6


def get_income_group_index(income):
    """
    See http://www.spainaccountants.com/rates.html
    0 : < 12450
    1 : < 20200
    2 : < 35200
    3 : < 60000
    4 : < 100000
    5 : < 150000
    6 : < 200000
    7 : < 250000
    8 : < 300000
    9 : > 300000
    """
    if income < 0:
        return -1
    elif income < 12450:
        return 0
    elif income < 20200:
        return 1
    elif income < 35200:
        return 2
    elif income < 60000:
        return 3
    elif income < 100000:
        return 4
    elif income < 150000:
        return 5
    elif income < 250000:
        return 6
    elif income < 300000:
        return 7
    elif income < 300000:
        return 8
    else:
        return 9
