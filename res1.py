#this python program will do MLR model with its assumptions and gives descriptive statistics

import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets, linear_model
import seaborn as sns
import statsmodels.api as sm

df=pd.read_excel(r'C:\Users\Shruti1\Desktop\py\excel234.xls')
#print(df)
vars =["CMS", "TINDF", "TOIRF", "TFRF", "TCBEF", "TLRF"]
vars1 =["TINDF", "TOIRF", "TFRF", "TCBEF", "TLRF"]
vars2=["CMS"]
df=df[vars]
df1=df[vars1].dropna()
df2=df[vars2].dropna()
print(df.head())
print(df1.head())


#Heat map for correlation matrix
correlation_matrix=df.corr().round(2)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(data=correlation_matrix,annot=True)
plt.show()

#normality of dependent variable
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(df["CMS"], bins = 20)
plt.show()

from statsmodels.graphics.regressionplots import plot_ccpr
from statsmodels.graphics.regressionplots import add_lowess

#MLR regression model with descriptive statistics
model = sm.OLS.from_formula("CMS ~ TINDF + TOIRF + TFRF+ TCBEF +TLRF", data=df)
result = model.fit()
print(result.summary())
print(df.describe())

#this qq plot checks residuals from our reg model are normaly distributed-qq plot of resudals to check normality
qq=sm.qqplot(result.resid,line='r')
plt.show()


#simple plot of residuals
stdres=pd.DataFrame(result.resid_pearson)
print(stdres)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
plt.show()


# leverage plot
livearge_plot=sm.graphics.influence_plot(result, size=8)
print(livearge_plot)


#final matches with spss multicollinearity assumption
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
X = add_constant(df1)
print('     VIF - TABLE\n','Variable    VIF\n',pd.Series([variance_inflation_factor(X.values, i)for i in range(X.shape[1])], index=X.columns))

#Plot for Homoscedasticity spss confirmed

fig, ax=plt.subplots(figsize=(8,3.5))
pred_val=result.fittedvalues.copy()
true_val=df['CMS'].values.copy()
resid=true_val-pred_val
store=sns.residplot(resid,pred_val)
#plt.title('Homoscedasticity')


#short cut to Homoscedasticity-to be checked resid value from above
plt.scatter(resid.index, resid.values)
plt.hlines(0,0,150)
plt.show()



#Linearity assumption trying
#Linearity assumption correct
from scipy.stats import zscore
pred_val=result.fittedvalues.copy()
z=zscore(pred_val)
print('zis....',z)
store1=plt.scatter(z,df2)
















