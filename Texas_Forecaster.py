def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

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

# AP = pd.read_csv('Total_AP.csv')
# AP['DistName'] = AP['DistName'].str.upper()
# AP.to_csv('Total_AP.csv', index=False)

### Fixing Wealth Per ADA ###
# wealth = pd.read_csv('Wealth_Original.csv')
# years = ['2011', '2012', '2013', '2014', '2015', '2016', '2017']
# yearly_wealth = []
# for year in years:
#     yearly_df = wealth[['DistName', year]]
#     yearly_df.columns = ['DistName', 'Wealth/ADA']
#     yearly_df['Year'] = int(year)
#     wanted_dists = []
#     for dist_name in list(yearly_df['DistName'].values):
#         if ' ISD' in dist_name:
#             wanted_dists.append(dist_name)
#     yearly_df = yearly_df.loc[yearly_df['DistName'].isin(wanted_dists)]
#     yearly_wealth.append(yearly_df)
# Total_wealth = pd.concat(yearly_wealth)

### Adding Regions to Wealth Per ADA ###
# SAT = pd.read_csv('Total_SAT.csv')
# SAT = SAT[['DistName', 'RegnName', 'Year']]
# Wealth = pd.read_csv('Total_Wealth.csv')
# new_wealth = pd.merge(Wealth, SAT, on=['DistName', 'Year'], how='inner')
# new_wealth = new_wealth.sort_values(['DistName', 'Year'])
# new_wealth.to_csv('Total_Wealth.csv', index=False)

### Merging SAT, ACT, AP, Enrollment, Wealth/ADA ###
# path = 'Total_*'
# file_names = glob.glob(path)
# dfs = [pd.read_csv(file) for file in file_names]
# Total_Merged = reduce(lambda x, y: pd.merge(x, y, on=['DistName', 'RegnName', 'Year'], how='inner'), dfs)
# Total_Merged = Total_Merged.sort_values(['DistName', 'Year'])
# Total_Merged.to_csv('Seven_Year_Historical.csv', index=False)

### Getting School Districts With the Full Seven Years of Data ###
# seven = pd.read_csv('Seven_Year_Historical.csv')
# seven = seven[seven.columns[:-2]]
# print(seven.head(5))
# seven = seven.dropna()
# wanted_dfs = []
# for dist in list(seven['DistName'].unique()):
#     dist_df = seven.loc[seven['DistName'] == dist]
#     if len(dist_df) == 7:
#         wanted_dfs.append(dist_df)
# new = pd.concat(wanted_dfs)
# new.to_csv('Seven_Year_Historical_new.csv', index=False)

### Forecasting the next 3 years (2018, 2019, 2020) ###
# historical = pd.read_csv('Seven_Year_Historical_new.csv')
# new_dfs = []
# for dist in list(historical['DistName'].unique()):
#     regn = list(historical.loc[historical['DistName'] == dist]['RegnName'].values)[0]
#     columns = ['ACT-Composite', 'ACT-Part_Rate', 'AP-11&12 Participating Students', 'AP-Total Exams', 'AP-Passed(%)',
#                'AP-Exams Taken Per Student', 'Enrolled 4-Year', 'Total Graduated', 'Enrolled 4-Year (%)', 'SAT-Total',
#                'SAT-Part_Rate', 'Wealth/ADA']
#     new_df = {'DistName': dist, 'RegnName': regn, 'Year': [2018, 2019, 2020]}
#     for col in columns:
#         dist_hist = historical.loc[historical['DistName'] == dist][[col, 'Year']]
#         X = dist_hist.drop(col, axis=1).values
#         y = dist_hist[col].values
#         reg = LinearRegression(fit_intercept=False)
#         reg.fit(X, y)
#         y_pred = reg.predict([[2018], [2019], [2020]])
#         new_df.update({col: y_pred})
#         predictions_df = pd.DataFrame(new_df)
#         hist = historical.loc[historical['DistName'] == dist][['DistName', 'RegnName', 'Year', 'ACT-Composite', 'ACT-Part_Rate', 'AP-11&12 Participating Students', 'AP-Total Exams', 'AP-Passed(%)',
#                'AP-Exams Taken Per Student', 'Enrolled 4-Year', 'Total Graduated', 'Enrolled 4-Year (%)', 'SAT-Total',
#                'SAT-Part_Rate', 'Wealth/ADA']]
#         if col == columns[-1]:
#             new_df_forecast = hist.append(predictions_df)
#             new_dfs.append(new_df_forecast)
# Hist_Forecast = pd.concat(new_dfs).sort_values(['DistName', 'Year'])
# Hist_Forecast.to_csv('Forecasted_Features.csv', index=False)

### Model for Predicting College Graduation ###
grad = pd.read_csv('Graduation_Historical.csv')
grad = grad[grad.columns[3:]]
X = grad.drop(['Graduated 4-Year (%)', 'Graduated 4-Year'], axis=1).values
y = grad['Graduated 4-Year (%)'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
for regression_model in [Ridge(), Lasso()]:
    params = {'alpha': [.0001, .001, .01, .1]}
    reg = GridSearchCV(regression_model, param_grid=params, cv=5)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    # print(reg.score(X_test, y_test))
    # print(reg.best_score_)
    # print(reg.best_params_)
    # print(reg.score(X_train, y_train))

### Merging Historical/Predicted Test Features with Historical College Grad ###
# years = [2011, 2012, 2013, 2014]
# forecast1 = pd.read_csv('Forecasted_Features.csv')
# forecast = forecast1.loc[forecast1['Year'].isin(years)]
# grad = pd.read_csv('Graduation_Historical.csv')
# grad = grad[['DistName', 'Graduated 4-Year (%)', 'Year']]
# new = pd.merge(grad, forecast, on=['DistName', 'Year'], how='inner')
# new2 = pd.merge(forecast1, new, on=['DistName', 'Year'], how='outer')
# new2 = new2.iloc[:, :16]
# new2 = new2[['DistName', 'RegnName_x', 'Year', 'ACT-Composite_x', 'ACT-Part_Rate_x', 'AP-11&12 Participating Students_x', 'AP-Total Exams_x', 'AP-Passed(%)_x', 'AP-Exams Taken Per Student_x',
# 'Enrolled 4-Year_x', 'Total Graduated_x', 'Enrolled 4-Year (%)_x', 'SAT-Total_x', 'SAT-Part_Rate_x', 'Wealth/ADA_x', 'Graduated 4-Year (%)']]
# new2.columns = ['DistName', 'RegnName', 'Year', 'ACT-Composite', 'ACT-Part_Rate', 'AP-11&12 Participating Students', 'AP-Total Exams', 'AP-Passed(%)', 'AP-Exams Taken Per Student',
# 'Enrolled 4-Year', 'Total Graduated', 'Enrolled 4-Year (%)', 'SAT-Total', 'SAT-Part_Rate', 'Wealth/ADA', 'Graduated 4-Year (%)']
# new2.to_csv('Pre_Forecasted_Graduation.csv', index=False)


### Predicting Missing College Graduation (%) ###
hist_forc = pd.read_csv('Pre_Forecasted_Graduation.csv')
hist_forc = hist_forc.fillna('NaN')
columns = list(hist_forc.columns[3:-1])
print(columns)
for idx, row in hist_forc.iterrows():
    if row['Graduated 4-Year (%)'] == 'NaN':
        features = [row[col] for col in columns]
        features = np.array(features).reshape(1, -1)
        hist_forc.loc[idx, 'Graduated 4-Year (%)'] = reg.predict(features)
hist_forc.to_csv('Final_Forecast.csv', index=False)