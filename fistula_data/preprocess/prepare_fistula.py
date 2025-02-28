"""
This script is used to prepare the dataset for training and test.

"""
# -*-coding:utf-8-*-
import os
import sys

import numpy as np

sys.path.extend(["../../", "../", "./"])
from commons.utils import *


data_root = "../../../medical_data/esophageal"
data_path = "../../../medical_data/esophageal/fistula"
csv_data_path = "../../../medical_data/esophageal/esophageal_fistula.csv"
MIN_IMG_BOUND = -200  # Everything below: Water  -62   -200
MAX_IMG_BOUND = 300  # Everything above corresponds to bones  238    200
MIN_MSK_BOUND = 0.0  # Everything above corresponds
MAX_MSK_BOUND = 2.0  # Everything above corresponds


def read_csv(csv_path):
    fistula_df = pd.read_csv(csv_path,
                             usecols=['name', '训练组a验证组b', 'Fistula', '序号', 'sex', 'age', 'ECOG PS', 'BMI',
                                      'History of Smoking', 'History of Drinking', 'History of hypertension',
                                      'History of diabetes', 'History of coronary heart disease (CHD)',
                                      'eating obstruction', 'T4', 'N2-3', 'M', 'Stage', 'Tumor site',
                                      'Longitudinal length of lesions(cm)', 'Pathology', 'General type',
                                      'chemotherapy', 'Taxol chemotherapy', 'Lines',
                                      'concurrent radiochemotherapy', 'Radiotherapy', 'Re-radiotherapy',
                                      'Total dose (Gy)', 'fraction', 'average single dose', 'technology',
                                      'Radiotherapy range (esophagus)',
                                      'Radiotherapy range (metastatic lymph nodes)',
                                      'Radiotherapy area (lymphatic drainage area)', 'Target therapy',
                                      'ALB（g/L）', 'CHO（mmol/L)'])
    # x, y = fistula_df['name'], fistula_df['grade']
    fistula_df['序号'] = fistula_df['序号'].map(lambda x: '{:g}'.format(x))
    fis_cat = pd.DataFrame(fistula_df, columns=['sex', 'ECOG PS', 'BMI',
                                                'History of Smoking', 'History of Drinking', 'History of hypertension',
                                                'History of diabetes', 'History of coronary heart disease (CHD)',
                                                'eating obstruction', 'T4', 'N2-3', 'M', 'Stage', 'Tumor site',
                                                'Pathology', 'General type',
                                                'chemotherapy', 'Taxol chemotherapy', 'Lines',
                                                'concurrent radiochemotherapy', 'Radiotherapy', 'Re-radiotherapy',
                                                'technology',
                                                'Radiotherapy range (esophagus)',
                                                'Radiotherapy range (metastatic lymph nodes)',
                                                'Radiotherapy area (lymphatic drainage area)', 'Target therapy',
                                                ])
    fis_num = pd.DataFrame(fistula_df, columns=[
        'age', 'Longitudinal length of lesions(cm)',
        'Total dose (Gy)', 'fraction', 'average single dose',
        'ALB（g/L）', 'CHO（mmol/L)'])
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    minus = lambda x: (x - 1) if np.min(x) > 0 else x
    for index, row in fis_num.items():
        fis_num[index] = fis_num[[index]].apply(max_min_scaler)
    np_num = fis_num.values
    for index, col in fis_cat.items():
        fis_cat[index] = fis_cat[[index]].apply(minus)
    cat_total = np.max(fis_cat.values, axis=0) + 1
    np_cat = np.zeros((fis_cat.values.shape[0], np.sum(cat_total))).astype(int)
    for index, row in fis_cat.iterrows():
        tmp = []
        for idx in range(len(row.values)):
            onehot = torch.zeros(cat_total[idx])
            tmp.append(np.array(onehot.scatter_(0, torch.tensor(row.values[idx]), 1).flatten()))
        np_cat[index] = np.concatenate(tmp).astype(int)
    fistula_df_training, fistula_df_test = [], []
    for index, row in fistula_df.iterrows():
        if row['训练组a验证组b'] == 'a':
            fistula_df_training.append({'name': row['name'],
                                        'Fistula': row['Fistula'],
                                        '序号': row['序号'],
                                        'num': np_num[index],
                                        'cat': np_cat[index]})
        else:
            fistula_df_test.append({'name': row['name'],
                                    'Fistula': row['Fistula'],
                                    '序号': row['序号'],
                                    'num': np_num[index],
                                    'cat': np_cat[index]})
    return fistula_df_training, fistula_df_test

