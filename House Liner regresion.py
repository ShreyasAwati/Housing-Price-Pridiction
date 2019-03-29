# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:50:05 2019

@author: Shreyas
"""

# import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report as cr
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
# pip install pandas_ml
from pandas_ml import ConfusionMatrix
import numpy as np
# from sklearn import cross_validation as cv
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score


# read the input file
# --------------------------------------
data = pd.read_csv("C:/Users/Shreyas/Downloads/train.csv")
data.head(5)

pd.set_option("display.expand_frame_repr", False)

# print the columns
# --------------------------------------
col = list(data.columns)
print(col)

# count of Rows and Columns 
# -----------------------------
data.shape

# total number of rows
# --------------------------------------
len(data.index)
data.describe(include=['object'])

#splitting y varable
target = data['SalePrice'] 
target.head()

import seaborn as sns
sns.distplot(target,hist=True)

target_log = np.log(target)

sns.distplot(target_log,hist=True)

import matplotlib
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"Sale Price":data["SalePrice"],"Log Sale Price ":target_log}) 
prices.hist()

#drop y varables

raw_data = data
data = data.drop(["SalePrice"], axis=1)
data.head()


#changing the data type
data['MSSubClass'] = data['MSSubClass'].apply(str)


data['OverallCond'] = data['OverallCond'].astype(str)


data['YrSold'] = data['YrSold'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)


#droping unwated coloumns
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF'] 
data = data.drop(["TotalBsmtSF"], axis=1)
data = data.drop(["1stFlrSF"], axis=1) 
data = data.drop(["2ndFlrSF"], axis=1) 
data = data.drop(["Id"], axis=1)
data.head()


#split data into catga and numirical
categorical_columns = [col for col in data.columns.values if data[col].dtype == 'object']


data_cat = data[categorical_columns]
#numeric varables in 
data_num = data.drop(categorical_columns, axis=1)
#describe the numiric data
data_num.describe()
data_num.shape
data_cat.shape
data_cat.head

#Skewness checking
data_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);

#skewnessnormalizaltion

from scipy.stats import skew
data_num_skew = data_num.apply(lambda x: skew(x.dropna())) 
data_num_skew = data_num_skew[data_num_skew > .75]


data_num[data_num_skew.index] = np.log1p(data_num[data_num_skew.index])

data_num_skew

#ploting normalized plot

data_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);

#Mena normailization
data_num = ((data_num - data_num.mean())/(data_num.max() - data_num.min())) 
data_num.describe() 

data_num.hist(figsize=(16, 20),xlabelsize=8, ylabelsize=8);


#analysing missing values
null_in_HousePrice = data.isnull().sum()
null_in_HousePrice = null_in_HousePrice[null_in_HousePrice > 0] 
null_in_HousePrice.sort_values(inplace=True) 
null_in_HousePrice.plot.bar()

#calculating percentage of null values

total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 
missing_data.head(15)


#Handling missing value

data_len = data_num.shape[0]


for col in data_num.columns.values: missing_values = data_num[col].isnull().sum()




if missing_values > 260:

    data_num = data_num.drop(col, axis = 1)

else:

    data_num = data_num.fillna(data_num[col].median())




#For cat varables
    
data_len = data_cat.shape[0]


for col in data_cat.columns.values: missing_values = data_cat[col].isnull().sum()

if missing_values > 50:

    print("droping column: {}".format(col)) 
    data_cat.drop(col, axis = 1)

else:

    pass
  
    
data_cat.describe()

#creating dummies varable 

data_cat.columns


data_cat_dummies= pd.get_dummies(data_cat,drop_first=True)



data_cat_dummies.head()

print("Numerical features : " + str(len(data_num.columns))) 
print("Categorical features : " + str(len(data_cat_dummies.columns)))

newdata = pd.concat([data_num, data_cat_dummies], axis=1)

#EDA

sns.factorplot("Fireplaces","SalePrice",data=raw_data,hue="FireplaceQu");
'''
If there are two ﬁreplaces, the Sales Price increases.
 Also, if there are ﬁreplace of Excellent quality in the 
 house the Sales Price increases.
 '''
FireplaceQu = raw_data["FireplaceQu"].fillna('None') 
pd.crosstab(raw_data.Fireplaces, raw_data.FireplaceQu)

#bar plot

sns.barplot(raw_data.OverallQual,raw_data.SalePrice)

#As we can see, the Sales Price increases with the increase in Overall Quality.

#ploting according to zons 

labels = raw_data["MSZoning"].unique()
sizes = raw_data["MSZoning"].value_counts().values 
explode=[0.1,0,0,0,0]
parcent = 100.*sizes/sizes.sum()
labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(labels, parcent)]

colors = ['yellowgreen', 'gold', 'lightblue', 'lightcoral','blue']
patches, texts= plt.pie(sizes, colors=colors,explode=explode,
                        shadow=True,startangle=90)
plt.legend(patches, labels, loc="best")

plt.title("Zoning Classification") 
plt.show()

sns.violinplot(raw_data.MSZoning,raw_data["SalePrice"])
plt.title("MSZoning wrt Sale Price")
plt.xlabel("MSZoning")
plt.ylabel("Sale Price");

#Sales square feet wise
SalePriceSF = raw_data['SalePrice']/raw_data['GrLivArea']
plt.hist(SalePriceSF, color="green")
plt.title("Sale Price per Square Foot") 
plt.ylabel('Number of Sales')
plt.xlabel('Price per square feet');

#most sale is around 100 and 150 square feet 

#sale price with years


ConstructionAge = raw_data['YrSold'] - raw_data['YearBuilt'] 
plt.scatter(ConstructionAge, SalePriceSF)
plt.ylabel('Price per square foot (in dollars)') 
plt.xlabel("Construction Age of house");

#sale price go down with age 

#sale price with cerntralair


sns.stripplot(x="HeatingQC", y="SalePrice",data=raw_data,hue='CentralAir',jitter=True,split=True) 
plt.title("Sale Price vs Heating Quality");

#the price increases rapdildly with AC

sns.boxplot(raw_data["FullBath"],raw_data["SalePrice"])
plt.title("Sale Price vs Full Bathrooms");

#Kitchen Grade VS Saleprice

sns.factorplot("KitchenAbvGr","SalePrice",data=raw_data,hue="KitchenQual")
plt.title("Sale Price vs Kitchen");

#having exclent quality of kitchen  hikes the price


##Corelation###


data_num.corr()

##Corelation plot##


import matplotlib.pyplot as plt 
corr=data_num.corr() 
plt.figure(figsize=(30, 30))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.5)], cmap='YlGnBu', vmax=1.0, vmin=-1.0, linewidths=0.1, annot=True, annot_kws={"size": 8}, square=True);
plt.title('Correlation between features')

#Linear Regrasion model


from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(newdata, target_log, test_size = 0.30)
print("x_train ",x_train.shape) 
print("x_test ",x_test.shape) 
print("y_train ",y_train.shape)
print("y_test ",y_test.shape)	

#building Base model


import statsmodels.api as sm

model1 = sm.OLS(y_train, x_train).fit()
model1.summary()

from IPython.display import Image

#RMSE
def rmse(predictions, targets):
    differences = predictions - targets 
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val

cols = ['Model', 'R-Squared Value', 'Adj.R-Squared Value', 'RMSE'] 
models_report = pd.DataFrame(columns = cols)

predictions1 = model1.predict(x_test)


tmp1 = pd.Series({'Model': " Base Linear Regression Model", 'R-Squared Value' : model1.rsquared,
'Adj.R-Squared Value': model1.rsquared_adj, 'RMSE': rmse(predictions1, y_test)})

model1_report = models_report.append(tmp1, ignore_index = True) 
model1_report


#bulding model with constant

df_constant = sm.add_constant(newdata)

x_train1,x_test1, y_train1, y_test1 = train_test_split(df_constant,target_log,test_size = 0.30)


import statsmodels.api as sm


model2 = sm.OLS(y_train1, x_train1).fit()

model2.summary()


#prediction
predictions2 = model2.predict(x_test1)

tmp2 = pd.Series({'Model': " Linear Regression Model with Constant", 'R-Squared Value' : model2.rsquared,
'Adj.R-Squared Value': model2.rsquared_adj, 'RMSE': rmse(predictions2, y_test1)})

model2_report = models_report.append(tmp2, ignore_index = True)
model2_report

#calaculating varaience

print ("\nVariance Inflation Factor")
cnames = x_train1.columns
for i in np.arange(0,len(cnames)): 
    xvars = list(cnames)
    yvar = xvars.pop(i)
    mod = sm.OLS(x_train1[yvar],(x_train1[xvars]))
    res = mod.fit()
    vif = 1/(1-res.rsquared) 
    print (yvar,round(vif,3))

#Removing varables above 100

vif_100 = ['MSSubClass_20','MSSubClass_60','RoofStyle_Gable','RoofStyle_Hip','RoofMatl_CompShg','Exte rior1st_MetalSd','Exterior1st_VinylSd','Exterior2nd_VinylSd','GarageQual_TA','GarageCond_TA']

to_keep = [x for x in x_train1 if x not in vif_100] 
x_train2 = x_train1[to_keep]
x_train2.head()

#model building removing VIF  above 100 

import statsmodels.api as sm



model3 = sm.OLS(y_train1,x_train2).fit()

model3.summary()
      