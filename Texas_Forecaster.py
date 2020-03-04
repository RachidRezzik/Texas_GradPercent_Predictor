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
import seaborn as sns
import matplotlib.pyplot as plt


#Setting Pandas DF options#
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

### Fixing College Enrollment DFs ###
# SAT = pd.read_csv('Total_SAT.csv')
# SAT = SAT[['DistName', 'RegnName', 'Year']]

### Fixing College Enrollment DFs ###
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


### SAT/ACT ###
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
file_names = ['Total_ACT.csv', 'Total_AP.csv', 'Total_Enrollment.csv', 'Total_SATwithout2017.csv', 'Total_Wealth.csv']
dfs = [pd.read_csv(file) for file in file_names]
Total_Merged = reduce(lambda x, y: pd.merge(x, y, on=['DistName', 'RegnName', 'Year'], how='inner'), dfs)
Total_Merged = Total_Merged.sort_values(['DistName', 'Year'])
wanted_dfs = []
for dist in list(Total_Merged['DistName'].unique()):
    dist_df = Total_Merged.loc[Total_Merged['DistName'] == dist]
    if len(dist_df) == 7:
        wanted_dfs.append(dist_df)
Total_Merged = pd.concat(wanted_dfs)
Total_Merged.to_csv('Seven_Year_Historical2.csv', index=False)

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
# grad = pd.read_csv('Graduation_Historical.csv')
# grad = grad[grad.columns[3:]]
# X = grad.drop(['Graduated 4-Year (%)', 'Graduated 4-Year'], axis=1).values
# y = grad['Graduated 4-Year (%)'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
# for regression_model in [Ridge(), Lasso()]:
#     params = {'alpha': [.0001, .001, .01, .1]}
#     reg = GridSearchCV(regression_model, param_grid=params, cv=5)
#     reg.fit(X_train, y_train)
#     y_pred = reg.predict(X_test)
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
# hist_forc = pd.read_csv('Pre_Forecasted_Graduation.csv')
# hist_forc = hist_forc.fillna('NaN')
# columns = list(hist_forc.columns[3:-1])
# print(columns)
# for idx, row in hist_forc.iterrows():
#     if row['Graduated 4-Year (%)'] == 'NaN':
#         features = [row[col] for col in columns]
#         features = np.array(features).reshape(1, -1)
#         hist_forc.loc[idx, 'Graduated 4-Year (%)'] = reg.predict(features)
# hist_forc.to_csv('Final_Forecast.csv', index=False)

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

#### TRENDS ####
# Wealth/ADA
# total = pd.read_csv('Final_Forecast.csv')
# years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
# total = total.loc[total['Year'].isin(years)]
# major_regions = ['Houston', 'San Antonio', 'Austin', 'Richardson', 'Fort Worth']
# color_dict = dict({'Houston':'blue',
#                   'San Antonio':'orange',
#                   'Austin': 'green',
#                   'Richardson': 'red',
#                    'Fort Worth': 'purple'})
# print(np.corrcoef(total['Wealth/ADA'], total['Graduated 4-Year (%)']))
# print(np.corrcoef(total['Wealth/ADA'], total['Enrolled 4-Year (%)']))
# # plt.subplot(2,1,1)
# graph = sns.lmplot(x='Wealth/ADA', y='Graduated 4-Year (%)', hue='RegnName', palette=color_dict, data=total, fit_reg=False)
# sns.regplot(x='Wealth/ADA', y='Graduated 4-Year (%)', data=total, scatter=False, ax=graph.axes[0, 0], line_kws={"color":"black"})
# plt.title('Graduated 4-Year (%) vs. Wealth/ADA')
# plt.xlabel('Wealth/ADA ($)')
# plt.show()

# graph = sns.lmplot(x='Wealth/ADA', y='Enrolled 4-Year (%)', hue='RegnName', palette=color_dict, data=total, fit_reg=False)
# sns.regplot(x='Wealth/ADA', y='Enrolled 4-Year (%)', data=total, scatter=False, ax=graph.axes[0, 0], line_kws={"color":"black"})
# plt.title('Enrolled 4-Year (%) vs. Wealth/ADA')
# plt.xlabel('Wealth/ADA ($)')
# plt.show()


### SAT and ACT participation, Regional trends, Distribution ###
# participation = pd.pivot_table(total, columns='Year', values=['SAT-Part_Rate', 'ACT-Part_Rate'], aggfunc=np.mean)
# for region in major_regions:
#     region_total = total.loc[total['RegnName'] == region]
#     sat_trend = pd.pivot_table(region_total, index='Year', values='AP-Passed(%)', aggfunc=np.mean)
#     sat_trend = pd.DataFrame(sat_trend.to_records())
#     plt.plot(sat_trend['Year'], sat_trend['AP-Passed(%)'], color=color_dict[region], marker='s', label=region)
# plt.xlabel('Year')
# plt.ylabel('AP-Passed (%)')
# plt.title('Regional Average Percentage of Passed AP Exams (2011 - 2017)')
# plt.legend(loc='center left', bbox_to_anchor=(1, .5))
# plt.show()
# plt.hist(total['AP-Passed(%)'], color='green', edgecolor='black')
# plt.title('AP-Passed (%) Distribution (Major Regions: 2011 - 2017)')
# plt.xlabel('AP-Passed (%)')
# plt.ylabel('Count')
# plt.show()



### AP Exams Taken Per Student | Wealth/ADA ###
# exam_student = pd.pivot_table(total, columns='Year', index='RegnName', values=['AP-Exams Taken Per Student'], aggfunc=np.mean)
# print(exam_student)
# print(np.corrcoef(total['Wealth/ADA'], total['AP-Exams Taken Per Student']))
# sns.regplot(data=total, x=total['Wealth/ADA'], y=total['AP-Exams Taken Per Student'], color='red', hue='RegnName')
# plt.show()
# graph = sns.lmplot(x='Wealth/ADA', y='AP-Exams Taken Per Student', hue='RegnName', palette=color_dict, data=total, fit_reg=False)
# sns.regplot(x='Wealth/ADA', y='AP-Exams Taken Per Student', data=total, scatter=False, ax=graph.axes[0, 0], line_kws={"color":"black"})
# plt.title('AP Exams Taken Per Student vs. Wealth/ADA')
# plt.xlabel('Wealth/ADA ($)')
# plt.show()

### Wealth/ADA ###
# wealth = pd.pivot_table(total, index='RegnName', values=['Wealth/ADA'], aggfunc=np.mean)
# print(wealth)


### Regional Grad % (Yearly) ###
# enrol_grad = pd.pivot_table(total, index=['Year', 'RegnName'], values='Graduated 4-Year (%)', aggfunc=np.mean)
# enrol_grad = enrol_grad.sort_values(['Year', 'Graduated 4-Year (%)'], ascending=[True, False])
# print(enrol_grad)

# enrol_grad = pd.pivot_table(total, index=['Year', 'RegnName'], values=['Enrolled 4-Year (%)', 'Wealth/ADA'], aggfunc=np.mean)
# enrol_grad = enrol_grad.sort_values(['Year', 'Wealth/ADA'], ascending=[True, False])
# print(enrol_grad)

