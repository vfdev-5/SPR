#
#
#


def to_yearmonth(yearmonthdate_str):
    """
    Convert '2016-01-23' -> 201601
    """
    # yearmonth = int(yearmonth_str[:7].replace('-', ''))
    yearmonth = int(yearmonthdate_str[:4] + yearmonthdate_str[5:7])
    return yearmonth


def to_ym_dec(ym):
    """
    XXXXYY -> XXXX.ZZ
    ZZ = (YY - 1) * 100.0 / 12.0
    """
    XXXX = int(ym * 0.01)
    YY = int(100 * (ym * 0.01 - XXXX) + 0.5)
    ZZ = (YY - 1) * 100.0 / 12.0
    ym_dec = XXXX + 0.01 * ZZ
    return ym_dec


def to_ym(ym_dec):
    """
    XXXX.ZZ -> XXXXYY
    """
    XXXX = int(ym_dec)
    ZZ = ym_dec - XXXX
    YY = int(ZZ * 12.0 + 0.5) + 1
    ym = XXXX * 100 + YY
    return ym


def to_nb_months(ym_dec):
    """
    XXXX.ZZ -> number of months
    """
    nb_years = int(ym_dec)
    zz = ym_dec - nb_years
    return 12 * nb_years + int(zz * 12.0 + 0.5)