import pandas as pd
import numpy as np
import glob
from functools import reduce
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#Setting Pandas DF options#
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

### Fixing College Enrollment DFs ###
# SAT = pd.read_csv('Total_SAT.csv')
# SAT = SAT[['DistName', 'RegnName', 'Year']]
# path = 'College_Enrollment/*'
# enrol_files = glob.glob(path)
# fixed_enrol_dfs = []
# for file_name in enrol_files:
#     file = pd.read_csv(file_name)
#     file = file[['DistName', 'Students Enrolled in Texas Public 4-year Universities', 'Total High School Graduates']]
#     file.columns = ['DistName', 'Enrolled 4-Year', 'Total Graduated']
#     col_list = file.columns[1:]
#     dist = list(file['DistName'])
#     dist = [str(i) for i in dist]
#     dist = [i[9:] for i in dist]
#     file['DistName'] = pd.Series(dist)
#     for col in col_list:
#         new_list = []
#         series_list = list(file[col].values)
#         for value in series_list:
#             if '*' in str(value):
#                 value = str(value).replace('*', '')
#             elif ',' in str(value):
#                 value = str(value).replace(',', '')
#             new_list.append(value)
#         file[col] = pd.Series(new_list)
#         file[col] = pd.to_numeric(file[col])
#     file['Enrolled 4-Year (%)'] = (file['Enrolled 4-Year'] / file['Total Graduated']) * 100
#     file['Year'] = int(file_name[19:23])
#     wanted_dists = []
#     for dist_name in list(file['DistName'].values):
#         if ' ISD' in dist_name:
#             wanted_dists.append(dist_name)
#     file = file.loc[file['DistName'].isin(wanted_dists)]
#     fixed_enrol_dfs.append(file)
# total_enrollment = pd.concat(fixed_enrol_dfs)
# test = pd.merge(SAT, total_enrollment, on=['DistName', 'Year'], how='inner')
# test = test.sort_values(['DistName', 'Year'])
# test.to_csv('Total_Enrollment.csv', index=False)

### SAT/ACT ###
# tests = ['SAT', 'ACT']
# major_regions = ['Houston', 'San Antonio', 'Austin', 'Richardson', 'Fort Worth']
# for test in tests:
#     fixed_test_dfs = []
#     path = test + '/*'
#     enrol_files = glob.glob(path)
#     for file_name in enrol_files:
#         if 'SAT' in file_name:
#             file = pd.read_csv(file_name)
#             file = file[['DistName', 'RegnName', 'Total', 'Part_Rate']]
#             file.columns = ['DistName', 'RegnName', 'SAT-Total', 'SAT-Part_Rate']
#         else:
#             file = pd.read_csv(file_name)
#             file = file[['DistName', 'RegnName', 'Compos', 'Part_Rate']]
#             file.columns = ['DistName', 'RegnName', 'ACT-Composite', 'ACT-Part_Rate']
#         wanted_dists = []
#         for dist_name in list(file['DistName'].values):
#             if ' ISD' in dist_name:
#                 wanted_dists.append(dist_name)
#         file = file.loc[(file['DistName'].isin(wanted_dists)) & (file['RegnName'].isin(major_regions))]
#         file['Year'] = int(file_name[4:8])
#         file['DistName'] = file['DistName'].str.upper()
#         fixed_test_dfs.append(file)
#     new_test_df = pd.concat(fixed_test_dfs)
#     new_test_df = new_test_df.dropna()
#     new_test_df.to_csv('Total_' + test + '.csv', index=False)

### AP ###
# fixed_AP_dfs = []
# path = 'AP/*'
# AP_files = glob.glob(path)
# for file_name in AP_files:
#     file = pd.read_csv(file_name)
#     cols = file.columns[7:]
#     for col in cols:
#         series_list = list(file[col].values)
#         value_list = []
#         for value in series_list:
#             if '<' in str(value):
#                 value = int(str(value).replace('<', '')) * (1 - .1)
#             elif ',' in str(value):
#                 value = int(str(value).replace(',', ''))
#             value_list.append(value)
#         file[col] = pd.Series(value_list)
#         file[col] = pd.to_numeric(file[col])
#     file = file[['DistName', 'RegnName', 'Exnees_Mskd', 'Exams_Mskd', 'Exams_Above_Crit_Rate']]
#     file.columns = ['DistName', 'RegnName', 'AP-11&12 Participating Students', 'AP-Total Exams', 'AP-Passed(%)']
#     file['AP-Exams Taken Per Student'] = file['AP-Total Exams'] / file['AP-11&12 Participating Students']
#     file['Year'] = int(file_name[3:7])
#     wanted_dists = []
#     for dist_name in list(file['DistName'].values):
#         if ' ISD' in dist_name:
#             wanted_dists.append(dist_name)
#     file = file.loc[(file['RegnName'].isin(major_regions)) & (file['DistName'].isin(wanted_dists)) & (file['AP-Total Exams'] > 50)]
#     fixed_AP_dfs.append(file)
# new_AP_df = pd.concat(fixed_AP_dfs).sort_values('Year')
# new_AP_df = new_AP_df.dropna()
# new_AP_df.to_csv('Total_AP.csv', index=False)

enrol = pd.read_csv('Total_Enrollment.csv')
print(enrol['DistName'].value_counts())