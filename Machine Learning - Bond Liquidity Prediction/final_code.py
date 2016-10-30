# importing various modules that would be required in the program

import pandas as pd
import numpy as np
import dateutil.parser as dateparser
from datetime import datetime
import dateutil.parser as dateparser
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer, LabelEncoder

# Preprocessing the ML_Bond_metadata.csv file after initial treatment in excel.
data = pd.read_csv('ML_Bond_metadata_corrected_dates.csv')
# print data.isnull().sum()

numerical_fields = [
    'coupon',
    'amtIssued',
    'amtOutstanding',
]

categorical_fields = [
    'issuer',
    'Market',
    'collateralType',
    'couponFrequency'
    'couponType',
    'industryGroup',
    'industrySector',
    'industrySubGroup',
    'maturityType',
    'securityType',
    'paymentRank',
    '144aflag',
    'ratingAgency1Rating',
    'ratingAgency2Rating',
    'ratingAgency1Watch',
    'ratingAgency2Watch'
]

date_fields = [
    'issueDate',
    'maturity',
    'ratingAgency1EffectiveDate',
    'ratingAgency2EffectiveDate'
]

# The difference in the amounts is a new feature added to the data to give better insights
data['AmtDiff'] = data['amtIssued'] - data['amtOutstanding']
# The duration between issue and maturity
data['DateDiff'] = data['maturity'] - data['IssueDate']

# Imputing values in the columns where NANs are found
for i in ['issueDate','maturity','ratingAgency1EffectiveDate','ratingAgency1EffectiveDate']:
    data[i] = data[i].fillna(data.median())

# Changing the value of couponFrequency from NAN to 0 if coupon is also zero
temp = []
for i in range(leng):
    if data['coupon'].iloc[i] == 0.00:
        temp.append(i)
for i in temp:
    j = j+1
    data.set_value(i,'couponFrequency',0)

# For Cleaning Categorical Data
for i in ['couponFrequency']:
    data[i] = LabelEncoder().fit_transform(data[i])

# For Cleaning Numeric Data
for i in ['AmtDiff','DateDiff','amtOutstanding', 'amtIssued']:
	data[i] = StandardScaler().fit_transform(data[i])

pd.to_csv('metadata_clean.csv')

# functions which will be used for cleaning the dataset.csv file
def get_cluster(x):
    if x == 0:
        return 'A'
    elif x == 1:
        return 'B'
    elif x == 2:
        return 'C'
    elif x == 3:
        return 'D'
    elif x == 4:
        return 'E'
    elif x == 5:
        return 'F'
    elif x == 6:
        return 'G'
    elif x == 7:
        return 'H'
    elif x == 8:
        return 'I'
    elif x == 9:
        return 'J'

# Starting of the training time i.e. 16 March 2016 is considered as the epoch
def get_days(x):
    dt = datetime.strptime(x, "%d%b%Y")
    epoch = datetime(2016,3,16)
    return int((dt-epoch).total_seconds()/86400)

def get_time(x):
    if x.find('pm') == -1 and x.find('am') == -1:
        dt = datetime.strptime(x[:26], "%a %d%b%Y %H:%M:%S.%f")
    else:
        dt = datetime.strptime(x, "%a %d%b%Y %I:%M:%S.%f %p")
    epoch = datetime(2016,3,16)
    diff = dt - epoch
    return int(diff.total_seconds())

def get_side_back(x):
    if x == 0:
        return 'S'
    elif x == 1:
        return 'B'

def get_isin(x):
    return int(x[4:])

def correct_time(x):
    return x[:9]+'20'+x[9:]

# Clustering of the Bonds
# importing the module containg the python implementation of k-ptototype algorithm

from kmodes import kprototypes

X = data.as_matrix()
kproto = kprototypes.KPrototypes(n_clusters=10, init='Cao', verbose=2)
clusters = kproto.fit_predict(X, categorical=[0, 2, 5, 6, 7, 8, 9, 10, 11 ,12, 13, 14, 15, 16, 18, 19])
# New column has been created in the dataset for clusters
data['cluster'] = clusters

# saving the results to a csv file
data.to_csv('metadata_clean_cluster_10.csv')

# creating a temporary dataframe
temp_df = pd.DataFrame(columns=['isin','cluster'])
temp_df['isin']  = temp_df['isin']
temp_df['cluster'] = temp_df['cluster']

data1 = pd.read_csv('dataset.csv')

# price culd not be modelled
data1 = data1.drop(['price'])

data1['isin'] = data1['isin'].apply(get_isin)
data1['time'] = data1['time'].apply(correct_time)
data1['time'] = data1['time'].apply(get_time)
data1['date'] = data1['date'].apply(get_days)

data1 = data1.drop(['time'],axis=1)

# sequence to compress the various trades in a day into one single entry
isins = data1['isin'].unique()
for i in isins:
    temp = data1[data1['isin'] == i]
    temp_dates = temp['date'].unique()
    for j in temp_dates:
        temp1 = temp[temp['date'] == j]
        temp_side = temp1['side'].unique()
        for k in temp_side:
            temp2 = temp1[temp1['side'] == k]
            sum_vol = temp2['volume'].sum()
            res1 = pd.DataFrame([[i,sum_vol,k,j]],columns=['isin','volume','side','date'])
            res = res.append(res1)

data1 = res

# merge tables to include cluster data in the dataset also
data1 = data1.merge(temp_df, on='isin', how='left')
data1.to_csv('dataset_clean_cluster10.csv')
data1['cluster'] = data1['cluster'].apply(get_cluster)

# dividing the data into training and validation test
data1['is_t_data'] = np.random.uniform(0,1,len(data))<=0.75
train, validate = data1[data1['is_t_data']==True], data1[data1['is_t_data']==False]

# for the purpose of validation tests
train = train.drop(['is_t_data'],axis=1)
validate = validate.drop(['is_t_data'],axis=1)

model = MixedLM.from_formula('volume ~ side + date + cluster', data = data1, re_formula = 'date', groups = train['isin'])
result = model.fit()
print result.summary()

# template was created for easy access of output parameters
out = pd.read_csv('output_template.csv')
out['dummy'] = 0
dummy, X = dmatrices('dummy ~ side + date - 1', data=out, return_type='dataframe')

values = model.predict(X)

# values are returned as an numpy ndarray and after careful addition in excel, the csv file is uploaded.
