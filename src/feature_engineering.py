"""
Módulo de Feature Engineering para FraudShield.
Contiene las funciones fit/transform usadas para procesar el dataset
de forma reproducible y sin data leakage.
"""

import pandas as pd
import re


def encontrar_top_optimo_fit(serie, umbral_ganancia_marginal=0.02, max_n=20, min_cobertura=0.5):
    """
    FASE FIT: encuentra el top N óptimo de una columna (basado en ganancia marginal)
    y devuelve la lista de categorías top a usar, sin transformar nada.
    """
    total_no_nulos = serie.notna().sum()
    conteos = serie.value_counts()
    cobertura_acumulada = (conteos.cumsum() / total_no_nulos)

    ganancia_marginal = cobertura_acumulada.diff().fillna(cobertura_acumulada.iloc[0])

    n_optimo = max_n
    for i, ganancia in enumerate(ganancia_marginal.head(max_n), start=1):
        if ganancia < umbral_ganancia_marginal and cobertura_acumulada.iloc[i - 1] >= min_cobertura:
            n_optimo = i
            break

    top_categorias = conteos.head(n_optimo).index.tolist()
    return top_categorias


def agrupar_transform(serie, top_categorias):
    """
    FASE TRANSFORM: aplica una lista de categorías top ya calculada
    para agrupar cualquier serie en 'top N + otros + sin_dato'.
    """
    def _agrupar(x):
        if pd.isna(x):
            return 'sin_dato'
        elif x in top_categorias:
            return x
        else:
            return 'otros'
    return serie.apply(_agrupar)


def limpiar_nombre_columna(col):
    """Reemplaza caracteres especiales en nombres de columna (requerido por LightGBM)."""
    return re.sub(r'[^A-Za-z0-9_]', '_', col)


def fit_feature_engineering(train_df, grupos_v):
    """
    Calcula todas las estadísticas de referencia necesarias para el
    Feature Engineering, usando ÚNICAMENTE el DataFrame de train recibido.
    """
    stats = {}

    v_cols = [col for col in train_df.columns if col.startswith('V') and 'is_missing' not in col]
    stats['v_cols'] = v_cols
    stats['medianas_v'] = train_df[v_cols].median()

    stats['top_p_email'] = train_df['P_emaildomain'].value_counts().head(5).index.tolist()
    stats['top_r_email'] = train_df['R_emaildomain'].value_counts().head(5).index.tolist()

    for col in ['DeviceInfo', 'id_30', 'id_31', 'id_33']:
        stats[f'top_{col}'] = encontrar_top_optimo_fit(train_df[col])

    stats['mediana_por_tarjeta'] = train_df.groupby('card1')['TransactionAmt'].median()

    return stats


def transform_feature_engineering(df_raw, stats, grupos_v):
    """
    FASE TRANSFORM: aplica todas las transformaciones de Feature Engineering
    a un DataFrame crudo, usando estadísticas ya calculadas.
    """
    df_out = df_raw.copy()

    for grp_name, cols in grupos_v.items():
        df_out[f'{grp_name}_is_missing'] = df_out[cols[0]].isnull().astype(int)

    df_out[stats['v_cols']] = df_out[stats['v_cols']].fillna(stats['medianas_v'])

    df_out['P_emaildomain_grouped'] = agrupar_transform(df_out['P_emaildomain'], stats['top_p_email'])
    df_out['R_emaildomain_grouped'] = agrupar_transform(df_out['R_emaildomain'], stats['top_r_email'])

    for col in ['DeviceInfo', 'id_30', 'id_31', 'id_33']:
        df_out[f'{col}_grouped'] = agrupar_transform(df_out[col], stats[f'top_{col}'])

    df_out['amt_relative_to_card'] = df_out['TransactionAmt'] / df_out['card1'].map(stats['mediana_por_tarjeta'])
    df_out['amt_relative_to_card'] = df_out['amt_relative_to_card'].fillna(1.0)

    return df_out


GRUPOS_V = {
    'v_grp1': ['V279', 'V280', 'V284', 'V285', 'V286', 'V287', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V297', 'V298', 'V299', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321'],
    'v_grp2': ['V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137'],
    'v_grp3': ['V281', 'V282', 'V283', 'V288', 'V289', 'V296', 'V300', 'V301', 'V313', 'V314', 'V315'],
    'v_grp4': ['V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34'],
    'v_grp5': ['V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74'],
    'v_grp6': ['V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94'],
    'v_grp7': ['V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52'],
    'v_grp8': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11'],
    'v_grp9': ['V220', 'V221', 'V222', 'V227', 'V234', 'V238', 'V239', 'V245', 'V250', 'V251', 'V255', 'V256', 'V259', 'V270', 'V271', 'V272'],
    'v_grp10': ['V169', 'V170', 'V171', 'V174', 'V175', 'V180', 'V184', 'V185', 'V188', 'V189', 'V194', 'V195', 'V197', 'V198', 'V200', 'V201', 'V208', 'V209', 'V210'],
    'v_grp11': ['V167', 'V168', 'V172', 'V173', 'V176', 'V177', 'V178', 'V179', 'V181', 'V182', 'V183', 'V186', 'V187', 'V190', 'V191', 'V192', 'V193', 'V196', 'V199', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V211', 'V212', 'V213', 'V214', 'V215', 'V216'],
    'v_grp12': ['V217', 'V218', 'V219', 'V223', 'V224', 'V225', 'V226', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V235', 'V236', 'V237', 'V240', 'V241', 'V242', 'V243', 'V244', 'V246', 'V247', 'V248', 'V249', 'V252', 'V253', 'V254', 'V257', 'V258', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278'],
    'v_grp13': ['V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339'],
    'v_grp14': ['V143', 'V144', 'V145', 'V150', 'V151', 'V152', 'V159', 'V160', 'V164', 'V165', 'V166'],
    'v_grp15': ['V138', 'V139', 'V140', 'V141', 'V142', 'V146', 'V147', 'V148', 'V149', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V161', 'V162', 'V163'],
}