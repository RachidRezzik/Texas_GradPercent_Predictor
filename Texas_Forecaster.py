def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
import glob
from functools import reduce
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, Normalizer, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

#Setting Pandas DF options#
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

###########___________________DATA WRANGLING__________________________________________________________##########

### Fixing College Enrollment DFs ###
# # SAT = pd.read_csv('Total_SAT.csv')
# # SAT = SAT[['DistName', 'RegnName', 'Year']]
#
# ### Fixing College Enrollment DFs ###
# path = 'College_Enrollment/*'
# enrol_files = glob.glob(path)
# fixed_enrol_dfs = []
# for file_name in enrol_files:
#     file = pd.read_csv(file_name)
#     file = file[['DistName', 'Students Enrolled in Texas Public 4-year Universities',
#                  'Total High School Graduates']]
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
# SAT = pd.read_csv('Total_SAT.csv')
# SAT = SAT[['DistName', 'RegnName', 'Year']]
# test = pd.merge(SAT, total_enrollment, on=['DistName', 'Year'], how='inner')
# test = test.sort_values(['DistName', 'Year'])
# test.to_csv('Total_Enrollment.csv', index=False)
#
#
# ### SAT/ACT ###
# tests = ['SAT', 'ACT']
# major_regions = ['Houston', 'San Antonio', 'Austin', 'Richardson', 'Fort Worth']
# for test in tests:
#     fixed_test_dfs = []
#     path = test + '/*'
#     enrol_files = glob.glob(path)
#     for file_name in enrol_files:
#         if file_name in enrol_files[:-1] and 'SAT' in file_name:
#             file = pd.read_csv(file_name)
#             file = file[['DistName', 'RegnName', 'Math', 'Reading', 'Writing', 'Part_Rate']]
#             file['SAT-Total'] = 0
#             file['ERW'] = 0
#             for idx, row in file.iterrows():
#                 if 300 < row['Math'] < 320:
#                     previous_score = file.loc[idx, 'Math']
#                     file.loc[idx, 'Math'] = previous_score + 50
#                 elif 320 <= row['Math'] < 480:
#                     previous_score = file.loc[idx, 'Math']
#                     file.loc[idx, 'Math'] = previous_score + 40
#                 elif 480 <= row['Math'] < 550:
#                     previous_score = file.loc[idx, 'Math']
#                     file.loc[idx, 'Math'] = previous_score + 30
#                 elif 550 <= row['Math'] < 660:
#                     previous_score = file.loc[idx, 'Math']
#                     file.loc[idx, 'Math'] = previous_score + 20
#                 elif 660 <= row['Math'] < 730:
#                     previous_score = file.loc[idx, 'Math']
#                     file.loc[idx, 'Math'] = previous_score + 30
#             for idx, row in file.iterrows():
#                 if 590 <= (row['Writing'] + row['Reading']) < 610:
#                     file.loc[idx, 'SAT-Total'] = 350 + row['Math']
#                 elif 610 <= (row['Writing'] + row['Reading']) < 640:
#                     file.loc[idx, 'SAT-Total'] = 360 + row['Math']
#                 elif 640 <= (row['Writing'] + row['Reading']) < 660:
#                     file.loc[idx, 'SAT-Total'] = 370 + row['Math']
#                 elif 660 <= (row['Writing'] + row['Reading']) < 840:
#                     first_score = 380
#                     scores = [660 + (i * 20) for i in range(10)]
#                     for previous, current in zip(scores, scores[1:]):
#                         if previous <= (row['Writing'] + row['Reading']) < current:
#                             file.loc[idx, 'SAT-Total'] = first_score + row['Math']
#                         first_score += 10
#                 elif 840 <= (row['Writing'] + row['Reading']) < 850:
#                     file.loc[idx, 'SAT-Total'] = 470 + row['Math']
#                 elif 850 <= (row['Writing'] + row['Reading']) < 930:
#                     first_score = 480
#                     scores = [850 + (i * 20) for i in range(5)]
#                     for previous, current in zip(scores, scores[1:]):
#                         if previous <= (row['Writing'] + row['Reading']) < current:
#                             file.loc[idx, 'SAT-Total'] = first_score + row['Math']
#                         first_score += 10
#                 elif 930 <= (row['Writing'] + row['Reading']) < 940:
#                     file.loc[idx, 'SAT-Total'] = 520 + row['Math']
#                 elif 940 <= (row['Writing'] + row['Reading']) < 1180:
#                     first_score = 530
#                     scores = [940 + (i * 20) for i in range(13)]
#                     for previous, current in zip(scores, scores[1:]):
#                         if previous <= (row['Writing'] + row['Reading']) < current:
#                             file.loc[idx, 'SAT-Total'] = first_score + row['Math']
#                         first_score += 10
#                 elif 1180 <= (row['Writing'] + row['Reading']) < 1210:
#                     file.loc[idx, 'SAT-Total'] = 650 + row['Math']
#                 elif 1210 <= (row['Writing'] + row['Reading']) < 1250:
#                     first_score = 660
#                     scores = [1210 + (i * 20) for i in range(3)]
#                     for previous, current in zip(scores, scores[1:]):
#                         if previous < (row['Writing'] + row['Reading']) < current:
#                             file.loc[idx, 'SAT-Total'] = first_score + row['Math']
#                         first_score += 10
#             file = file[['DistName', 'RegnName', 'SAT-Total', 'Part_Rate']]
#             file.columns = ['DistName', 'RegnName', 'SAT-Total', 'SAT-Part_Rate']
#         elif file_name == enrol_files[-1] and 'SAT' in file_name:
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
#     new_test_df.to_csv('Total_' + test + 'without2017.csv', index=False)
#
# ### AP ###
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
# #     file = file[['DistName', 'RegnName', 'Exnees_Mskd', 'Exams_Mskd', 'Exams_Above_Crit_Rate']]
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
# # new_AP_df.to_csv('Total_AP.csv', index=False)
#
# # AP = pd.read_csv('Total_AP.csv')
# AP['DistName'] = AP['DistName'].str.upper()
# AP.to_csv('Total_AP.csv', index=False)
#
# ### Fixing Wealth Per ADA ###
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
#
# ### Adding Regions to Wealth Per ADA ###
# # SAT = pd.read_csv('Total_SAT.csv')
# # SAT = SAT[['DistName', 'RegnName', 'Year']]
# # Wealth = pd.read_csv('Total_Wealth.csv')
# new_wealth = pd.merge(Wealth, SAT, on=['DistName', 'Year'], how='inner')
# new_wealth = new_wealth.sort_values(['DistName', 'Year'])
# new_wealth.to_csv('Total_Wealth.csv', index=False)
#
### Merging SAT, ACT, AP, Enrollment, Wealth/ADA ###
# path = 'Total_*'
# file_names = glob.glob(path)
# dfs = [pd.read_csv(file) for file in file_names]
# Total_Merged = reduce(lambda x, y: pd.merge(x, y, on=['DistName', 'RegnName', 'Year'], how='inner'), dfs)
# Total_Merged = Total_Merged.sort_values(['DistName', 'Year'])
# wanted_dfs = []
# for dist in list(Total_Merged['DistName'].unique()):
#     dist_df = Total_Merged.loc[Total_Merged['DistName'] == dist]
#     if len(dist_df) == 7:
#         wanted_dfs.append(dist_df)
# Total_Merged = pd.concat(wanted_dfs)
# print(Total_Merged.head(20))
# Total_Merged.to_csv('Feature_Target_Data2.csv', index=False)


### Fixing Feature_Target_Data to include missing Wealth/ADA ####
f_t_data = pd.read_csv('Feature_Target_Data.csv')
wealth = pd.read_csv('Total_Wealth.csv')
new = pd.merge(f_t_data, wealth, on=['DistName', 'RegnName', 'Year'], how='inner')
new.to_csv('Feature_Target_Data2.csv', index=False)
##########___________________________DATA ANALYSIS___________________________________________________##############

#### TRENDS ####
# total = pd.read_csv('Feature_Target_Data.csv')
# years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
# years2 = [2011, 2012, 2013, 2014]
# total = total.loc[total['Year'].isin(years)]
# total2 = total.loc[total['Year'].isin([2014, 2015, 2016, 2017, 2018, 2019, 2020])]
# major_regions = ['Houston', 'San Antonio', 'Austin', 'Richardson', 'Fort Worth']
# color_dict = dict({'Houston':'blue',
#                   'San Antonio':'orange',
#                   'Austin': 'green',
#                   'Richardson': 'red',
#                    'Fort Worth': 'purple'})
#
# ### Graph Showing Regional College Graduation Trends + Forecast (2018) ###
# for region in major_regions:
#     region_total = total1.loc[total['RegnName'] == region]
#     sat_trend = pd.pivot_table(region_total, index='Year', values='Graduated 4-Year (%)', aggfunc=np.mean)
#     sat_trend = pd.DataFrame(sat_trend.to_records())
#     plt.plot(sat_trend['Year'], sat_trend['Graduated 4-Year (%)'], color=color_dict[region], marker='s', label=region)
# # for region in major_regions:
# #     region_total = total2.loc[total['RegnName'] == region]
# #     sat_trend = pd.pivot_table(region_total, index='Year', values='Graduated 4-Year (%)', aggfunc=np.mean)
# #     sat_trend = pd.DataFrame(sat_trend.to_records())
# #     plt.plot(sat_trend['Year'], sat_trend['Graduated 4-Year (%)'], color=color_dict[region], marker='s', linestyle='--', label=region)
# #
# # plt.xlabel('Year')
# # plt.ylabel('Graduated 4-Year (%)')
# # plt.title('Regional Average College Graduation (Hist:2011 - 2014, Forc:2018)')
# # plt.xticks([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018])
# # plt.show()
#
#### Graduated 4-Year vs Wealth/ADA #####

# # graph = sns.lmplot(x='Wealth/ADA', y='Graduated 4-Year (%)', hue='RegnName', palette=color_dict, data=total, fit_reg=False)
# # sns.regplot(x='Wealth/ADA', y='Graduated 4-Year (%)', data=total, scatter=False, ax=graph.axes[0, 0], line_kws={"color":"black"})
# # plt.title('Graduated 4-Year (%) vs. Wealth/ADA (2011 - 2014)')
# # plt.xlabel('Wealth/ADA ($)')
# # plt.show()
# print(np.corrcoef(total['Wealth/ADA'], total['Graduated 4-Year (%)']))

#### College Enrollment vs. Wealth/ADA ####
# # graph = sns.lmplot(x='Wealth/ADA', y='Enrolled 4-Year (%)', hue='RegnName', palette=color_dict, data=total, fit_reg=False)
# # sns.regplot(x='Wealth/ADA', y='Enrolled 4-Year (%)', data=total, scatter=False, ax=graph.axes[0, 0], line_kws={"color":"black"})
# # plt.title('Enrolled 4-Year (%) vs. Wealth/ADA')
# # plt.xlabel('Wealth/ADA ($)')
# # plt.show()
# # print(np.corrcoef(total['Wealth/ADA'], total['Enrolled 4-Year (%)']))
#
#
# ### Regional trends and Distribution ###

# # for region in major_regions:
# #     region_total = total.loc[total['RegnName'] == region]
# #     sat_trend = pd.pivot_table(region_total, index='Year', values='Wealth/ADA', aggfunc=np.mean)
# #     sat_trend = pd.DataFrame(sat_trend.to_records())
# #     plt.plot(sat_trend['Year'], sat_trend['Wealth/ADA'], color=color_dict[region], marker='s', label=region)
# plt.xlabel('Year')
# plt.ylabel('Wealth/ADA ($)')
# plt.title('Regional Average Wealth/ADA (2011 - 2017)')
# # plt.legend(loc='center left', bbox_to_anchor=(1, .5))
# plt.show()
# # plt.hist(total['SAT-Total'], color='red', edgecolor='black')
# # plt.title('SAT-Total Distribution (Major Regions: 2011 - 2017)')
# # plt.xlabel('SAT-Total')
# # plt.ylabel('Count')
# # plt.show()

##### SAT/ACT Participation #####
# participation = pd.pivot_table(total, columns='Year', values=['SAT-Part_Rate', 'ACT-Part_Rate'], aggfunc=np.mean)
# graph = sns.lmplot(x='ACT-Part_Rate', y='Enrolled 4-Year (%)', hue='RegnName', palette=color_dict, data=total, fit_reg=False)
# sns.regplot(x='ACT-Part_Rate', y='Enrolled 4-Year (%)', data=total, scatter=False, ax=graph.axes[0, 0], line_kws={"color":"black"})
# plt.title('Enrolled 4-Year (%) vs. ACT Participation Percentage')
# plt.xlabel('ACT-Part_Rate (%)')
# plt.show()
# print(np.corrcoef(total['ACT-Part_Rate'], total['Enrolled 4-Year (%)']))

# ### AP Exams Taken Per Student | Wealth/ADA ###
# # exam_student = pd.pivot_table(total, columns='Year', index='RegnName', values=['AP-Exams Taken Per Student'], aggfunc=np.mean)
# # print(exam_student)
# # print(np.corrcoef(total1['AP-Exams Taken Per Student'], total1['Wealth/ADA']))
# # print(np.corrcoef(total['Wealth/ADA'], total['AP-Exams Taken Per Student']))
# # sns.regplot(data=total, x=total['Wealth/ADA'], y=total['AP-Exams Taken Per Student'], color='red', hue='RegnName')
# # plt.show()
# # graph = sns.lmplot(x='Wealth/ADA', y='AP-Exams Taken Per Student', hue='RegnName', palette=color_dict, data=total, fit_reg=False)
# # sns.regplot(x='Wealth/ADA', y='AP-Exams Taken Per Student', data=total, scatter=False, ax=graph.axes[0, 0], line_kws={"color":"black"})
# # plt.title('AP Exams Taken Per Student vs. Wealth/ADA')
# # plt.xlabel('Wealth/ADA ($)')
# # plt.show()

# ### Wealth/ADA ###
# # wealth = pd.pivot_table(total, index='RegnName', values=['Graduated 4-Year (%)'], aggfunc=np.mean)
# # print(wealth.sort_values('Graduated 4-Year (%)', ascending=False))

# # plt.hist(total['Wealth/ADA'], color='purple', edgecolor='black')
# # plt.title('Wealth/ADA Distribution (Major Regions: 2011 - 2017)')
# # plt.xlabel('Wealth/ADA ($)')
# # plt.ylabel('Count')
# # plt.show()

#### Identify Highly Correlated Features ###
# grad = pd.read_csv('Feature_Target_Data.csv')
# years = [2011, 2012, 2013, 2014]
# grad = grad.loc[grad['Year'].isin(years)]
# grad = grad[grad.columns[3:-1]]
# corr_matrix = grad.corr().abs()
# print(corr_matrix)
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# print(upper)
# to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
# print(to_drop)



# ### VIF multicollinearity ###
# vif = pd.DataFrame()
# vif["VIF Factor"] = [variance_inflation_factor(grad_all_features.values, i) for i in range(grad_all_features.shape[1])]
# vif["features"] = grad_all_features.columns
# print(vif)
# vif2 = grad_all_features
# vif_cor = grad_all_features.corr()
# # new_vif = pd.DataFrame(np.linalg.inv(vif2.corr().values), index=vif_cor.index, columns=vif_cor.columns)
# # print(new_vif)
# vifs = pd.Series(np.linalg.inv(vif2.corr().values).diagonal(), index=vif_cor.index)
# print(grad_all_features.corr())


#########_________________________MACHINE LEARNING___________________________________________________###############

### Forecasting the next year (2018) ###
# historical = pd.read_csv('Feature_Target_Data.csv')
# new_dfs = []
# for dist in list(historical['DistName'].unique()):
#     regn = list(historical.loc[historical['DistName'] == dist]['RegnName'].values)[0]
#     columns = ['ACT-Composite', 'ACT-Part_Rate', 'AP-11&12 Participating Students', 'AP-Total Exams', 'AP-Passed(%)',
#                'AP-Exams Taken Per Student', 'Enrolled 4-Year', 'Total Graduated', 'Enrolled 4-Year (%)', 'SAT-Total',
#                'SAT-Part_Rate', 'Wealth/ADA']
#     new_df = {'DistName': dist, 'RegnName': regn, 'Year': [2018]}
#     for col in columns:
#         dist_hist = historical.loc[historical['DistName'] == dist][[col, 'Year']]
#         X = dist_hist.drop(col, axis=1).values
#         y = dist_hist[col].values
#         reg = LinearRegression(fit_intercept=False)
#         reg.fit(X, y)
#         y_pred = reg.predict([2018])
#         new_df.update({col: y_pred})
#         predictions_df = pd.DataFrame(new_df)
#         hist = historical.loc[historical['DistName'] == dist][['DistName', 'RegnName', 'Year', 'ACT-Composite', 'ACT-Part_Rate', 'AP-11&12 Participating Students', 'AP-Total Exams', 'AP-Passed(%)',
#                'AP-Exams Taken Per Student', 'Enrolled 4-Year', 'Total Graduated', 'Enrolled 4-Year (%)', 'SAT-Total',
#                'SAT-Part_Rate', 'Wealth/ADA']]
#         if col == columns[-1]:
#             new_df_forecast = hist.append(predictions_df)
#             new_dfs.append(new_df_forecast)
# Hist_Forecast = pd.concat(new_dfs).sort_values(['DistName', 'Year'])
# Hist_Forecast.to_csv('Forecasted_Features2.csv', index=False)


########## Model for Predicting College Graduation ############

# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

### Linear Regression ###
# grad = pd.read_csv('Feature_Target_Data.csv')
# years = [2011, 2012, 2013, 2014]
# grad = grad.loc[grad['Year'].isin(years)]
# grad = grad[grad.columns[3:]]
# X = grad.drop('Graduated 4-Year (%)', axis=1).values
# y = grad['Graduated 4-Year (%)'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
# linear_reg = LinearRegression()
# linear_reg.fit(X_train, y_train)
# y_pred_test = linear_reg.predict(X_test)
# y_pred_train = linear_reg.predict(X_train)
# print('Training RMSE:', np.sqrt(mean_squared_error(y_train, y_pred_train)), '\nTesting RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_test)))
# print('\nTraining R2: ', linear_reg.score(X_train, y_train), '\nTesting R2: ', linear_reg.score(X_test, y_test))
# print('\nTraining MAPE: ', mean_absolute_percentage_error(y_train, y_pred_train), '\nTesting MAPE: ', mean_absolute_percentage_error(y_test, y_pred_test))
# print(linear_reg.coef_)
# ### Interpreting Coefficients, Stats Model OLS ###
# from statsmodels.api import OLS
# X = grad[grad.columns[:-1]]
# y = grad[grad.columns[-1]]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
# olsreg = OLS(y_train, X_train)
# olsreg = olsreg.fit()
# print(olsreg.summary())

#
# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#
# for test_size in [.275, .25, .225, .2]:
#     rmse_all = []
#     r2_all = []
#     mape_all = []
#     rmse_drop = []
#     r2_drop = []
#     mape_drop = []
#     print('\n***Test Size: ', test_size, '***')
#     for i in range(1000):
#         grad = pd.read_csv('Feature_Target_Data.csv')
#         years = [2011, 2012, 2013, 2014]
#         grad = grad.loc[grad['Year'].isin(years)]
#         grad = grad[grad.columns[3:]]
#         X = grad.drop('Graduated 4-Year (%)', axis=1).values
#         y = grad['Graduated 4-Year (%)'].values
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
#         linear_reg = LinearRegression()
#         linear_reg.fit(X_train, y_train)
#         y_pred_test = linear_reg.predict(X_test)
#         y_pred_train = linear_reg.predict(X_train)
#         rmse_all.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
#         r2_all.append(linear_reg.score(X_test, y_test))
#         mape_all.append(mean_absolute_percentage_error(y_test, y_pred_test))
#         r2_max_index = r2_all.index(np.max(r2_all)
#
# ### After Dropping Features Causing Mulicollinearity ###
#         X = grad.drop(['Graduated 4-Year (%)', 'AP-Total Exams', 'Enrolled 4-Year', 'Total Graduated', 'AP-11&12 Participating Students', 'SAT-Total', 'Wealth/ADA'], axis=1).values
#         y = grad['Graduated 4-Year (%)'].values
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
#         linear_reg = LinearRegression()
#         linear_reg.fit(X_train, y_train)
#         y_pred_test = linear_reg.predict(X_test)
#         y_pred_train = linear_reg.predict(X_train)
#         rmse_drop.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
#         r2_drop.append(linear_reg.score(X_test, y_test))
#         mape_drop.append(mean_absolute_percentage_error(y_test, y_pred_test))
#     print('All Features: Performance on Test Set')
#     print('RMSE: ', np.mean(rmse_all), 'r2: ', np.mean(r2_all), 'MAPE: ', np.mean(mape_all))
#     print('\nAfter Dropping Features: Performance on Test Set')
#     print('RMSE: ', np.mean(rmse_drop), 'r2: ', np.mean(r2_drop), 'MAPE: ', np.mean(mape_drop))
#     print('Best r2: ', r2_all[r2_max_index], 'Random State: ', r2_max_index)









# cv_scores_linreg_r2 = cross_val_score(linear_reg, X_train, y_train, cv=5)
# cv_scores_linreg_mse = cross_val_score(linear_reg, X_train, y_train, cv=5, scoring=make_scorer(mean_squared_error))
# print("R^2 (testing): {}".format(linear_reg.score(X_test, y_test)))
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("RMSE (testing): {}".format(rmse))
# print("Mean R2 5-Fold CV Score (training): {}".format(np.mean(cv_scores_linreg_r2)))
# cv_scores_linreg_mse = [np.sqrt(i) for i in cv_scores_linreg_mse]
# print("Mean RMSE 5-Fold CV Score (training): {}".format(np.mean(cv_scores_linreg_mse)))
# print(cv_scores_linreg_r2)
# print(cv_scores_linreg_mse)

# for regression_model in [Ridge(), Lasso(), ElasticNet()]:
#     print('\n' + str(regression_model))
#     for scaler in [RobustScaler(), MinMaxScaler(), Normalizer(), StandardScaler()]:
#         steps = [('scaler', scaler), ('reg', regression_model)]
#         pipeline = Pipeline(steps)
#         params = {'reg__alpha': [.00001, .0001, .001, .01, .1]}
#         reg = GridSearchCV(pipeline, param_grid=params, scoring=make_scorer(mean_squared_error), cv=5)
#         reg.fit(X_train, y_train)
#         y_pred = reg.predict(X_test)
#         print('\n' + str(scaler))
#         print(np.sqrt(reg.best_score_), reg.best_params_, np.sqrt(reg.score(X_test, y_test)))
# scaler = RobustScaler()
# reg = ElasticNet(alpha=.0001)
# pipeline = make_pipeline(scaler, reg)
# reg.fit(X_train, y_train)
# y_pred = reg.predict(X_test)


# ### Predicting Missing College Graduation (%) ###
# hist_forc = pd.read_csv('Pre_Forecasted_Graduation2.csv')
# hist_forc = hist_forc.fillna('NaN')
# columns = list(hist_forc.columns[3:-1])
# print(columns)
# for idx, row in hist_forc.iterrows():
#     if row['Graduated 4-Year (%)'] == 'NaN':
#         features = [row[col] for col in columns]
#         features = np.array(features).reshape(1, -1)
#         hist_forc.loc[idx, 'Graduated 4-Year (%)'] = reg.predict(features)
# hist_forc.to_csv('Final_Forecast2.csv', index=False)

### Function to provide top # of options for year and region ###
# def top_5_options(year, region, options):
#     total = pd.read_csv('Final_Forecast.csv')
#     year_region = total.loc[(total['Year'] == year) & (total['RegnName'] == region)][['DistName', 'RegnName', 'Year', 'Graduated 4-Year (%)']]
#     year_region = year_region.sort_values('Graduated 4-Year (%)', ascending=False).reset_index(drop=True)
#     print(year_region.head(options))
#
# top_5_options(2020, 'Richardson', 6)

### Function to look up forecast for specific district ###
# def district_forecast(district, year):
#     total = pd.read_csv('Final_Forecast.csv')
#     district = total.loc[(total['DistName'] == district) & (total['Year'] == year)][['DistName', 'RegnName', 'Year', 'Graduated 4-Year (%)']].reset_index(drop=True)
#     print(district)
#
# district_forecast('AUSTIN ISD', 2018)

### New school district (has less than 7 years in existance, predict college grad % for students from certain class based on their features ###
# def predict_col_grad(feature_list):
#     district_features = np.array(feature_list).reshape(1, -1)
#     forecasted_grad = reg.predict(district_features)
#     print(forecasted_grad)
#
# predict_col_grad([24.4, 47, 350, 670, 73, 2.6, 70, 380, 20, 1108, 75, 700000])


