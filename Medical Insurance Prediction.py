#!/usr/bin/env python
# coding: utf-8

# #  ------------------ MEDICAL INSURANCE CHARGES PREDICTION -----------------------
Libraries Used :
    PANDAS, NUMPY, MATPLOTLIB, SEABORN, SKLEARN ,PICKLE
# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle

IMPORTING LIBRARIES
# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#  READING CSV FILE :
insurance = pd.read_csv("insurance.csv")


#                                                    DATA UNDERSTANDING

# In[4]:


# GETTING FIRST 10 RECORDS :
insurance.tail(10)


# In[5]:


# RETRIVING SUMMARY OF DATA :
insurance.info()


# In[6]:


# COUNT OF NUMBER OF ROWS :
print('----------------------------- ')
print("    TOTAL NUMBER OF ROWS     " )
print('----------------------------- ')
print("          ",insurance.shape[0])


# In[7]:


# COUNT OF NUMBER OF COLUMNS :
print('----------------------------- ')
print("   TOTAL NUMBER OF COLUMNS     " )
print('----------------------------- ')
print("             ",insurance.shape[1])


# In[8]:


# RETRIVING STATISTICAL SUMMARY OF DATA :
print('-------------------------------------------------------- ')
print("                 STATISTICAL SUMMARY                     " )
print('-------------------------------------------------------- ')
print(insurance.describe())


#                                            DATA PROCESSING / EDA

# In[9]:


# CHECKING THE ZERO VALUES :
print(' ------------------------')
print("       ZERO VALUES       ")
print(' ------------------------')
(insurance==0).sum()


# Here the children is showing zero values as 574 .
# But the 0 values in childrens column represents the Beneficiary is having no Childrens.
# Therefore, it is not a zero or missing value.
# So we canot adjust/fill or manipulate the value.

# In[10]:


# CHECKING THE MISSING VALUES :
print(' ------------------------')
print("       MISSING VALUES       ")
print(' ------------------------')
insurance.isnull().sum()

The data does not contains any missing or NaN values present.
Therefore, the is no need of adjusting/fillingg or manipulating the data.
#               DATA CLEANING / DATA PROCESSING / EDA  process has been successfully completed !

#                                        VISUAL DATA ANALYSIS ( VDA )

# In[11]:


# PLOTTING A SCATTER PLOT :

plt.figure(figsize=(12,8))
sns.lmplot(x='age', y='charges', data=insurance, fit_reg=False, hue='smoker', palette='Set2', legend=False)
plt.xlabel('AGE OF PERSON')
plt.ylabel('INSURANCE CHARGES')
plt.title('---------------------------------------------------------------------\n Comparing AGE and INSURANCE CHARGES \n considering Smoker/Non-Smoker \n ---------------------------------------------------------------------')
plt.legend(bbox_to_anchor=(1.02, 0.15), loc='center left' , title ='Smokers')
plt.show()


# In[ ]:





# In[12]:


# PlOTTING A BAR PLOT :

plt.figure(figsize=(10,8))
sns.countplot(x='sex', data=insurance, hue='region',palette='Set2')
plt.xlabel('Gender')
plt.ylabel('COUNT')
plt.title(" ------------------------------------------------ \n Count of Person based on \n  GENDER and REGION \n ------------------------------------------------ ")
plt.legend(bbox_to_anchor=(1.02, 0.15), loc='center left' , title ='Region')
plt.show()


# In[13]:


# PLOTTING A PIE CHART :

plt.figure(figsize=(13,8))
size=insurance['children'].value_counts()
labels=np.unique(insurance.children)
colors=['orange','pink','skyblue','palegreen','plum','lightgray']
explode=(0.05,0,0,0,0,0.01)
plt.pie(size, labels = labels,autopct='%1.1f%%', colors=colors , explode=explode)
plt.title("------------------------------------ \n COUNT OF CHILDREN \n ------------------------------------")
plt.legend(bbox_to_anchor=(1.02, 0.15), loc='center left' , title ='No.of Children')


# In[14]:


# PLOTTING BOX PLOT :

plt.figure(figsize=(12,6))
sns.boxplot(x="sex",y=insurance["bmi"], hue= 'smoker', data=insurance ,palette='Set3' )
plt.title("------------------------------------------------------------------------------ \n Comparing BMI with GENDER considering SMOKERS \n ----------------------------------------------------------------------------------")
plt.xlabel("GENDER")
plt.ylabel("BODY MASS INDEX")
plt.legend(bbox_to_anchor=(1.02, 0.15), loc='center left' , title ='Smoker')


# GROUPING OF CHILDRENS INTO FAMILY SIZE SO AS TO PLOT THE CHART ACCORDINGLY

# In[15]:


# GROUPING AND PLOTTING :

plt.figure(figsize=(13,6))
bins=[1,2,3,4]
labels = ['Small_Family','Medium_Family','Large_Family']
insurance['children_family'] = pd.cut(insurance['children'],bins=bins, labels=labels)
sns.boxplot(x ='children_family',y='charges', data = insurance , palette ='Set2')
plt.title(" --------------------------------------------------------------------------- \n Comparing FAMILY TYPE based on INSURANCE CHARGES  \n ---------------------------------------------------------------------------")
plt.xlabel("FAMILY SIZE DEPENDED ON CHILDREN")
plt.ylabel(" INSURANCE CHARGES")


# In[16]:


# PLOTTTING A HEAT MAP :

plt.figure(figsize=(7,5))

cmap="tab20"
center=0
annot= True

sns.heatmap(insurance.corr(),cmap=cmap,annot=annot)
plt.title("------------------------------------------------------\n Correlation of Numerical Variables \n ------------------------------------------------------")
plt.show()

Here the Heat Map represents numerical correlation values only.
#                                            KNOW MORE ABOUT DATA 

# In[17]:


# GETTING COLUMN NAME :

print('---------------------------------------')
print('           UNIQUE COLUMN NAMES         ')
print('---------------------------------------')
print(insurance.columns)


# In[18]:


# DROPPING THE CHILDREN_FAMILY COLUMN :

print('---------------------------------------')
print('           ADJUSTED COLUMN        ')
print('---------------------------------------')
insurance.drop('children_family', axis=1, inplace=True)
print(insurance.columns)


# In[19]:


# GETTING DATA TYPE OF COLUMN :
print('---------------------------------------')
print('           DATA TYPE OF COLUMN        ')
print('---------------------------------------')
print( insurance.dtypes)

Here age, bmi , charges is numerical type we do not need to manipulate it.
But sex, smoker , region are of category type we need to make further changes so to proceed with model building.
# In[20]:


# GETTING THE COUNT OF OBJECT DATA TYPE COLUMN :

print('--------------------------------------------')
print('           COUNT OF OBJECT DATA TYPE          ')
print('--------------------------------------------')
for col in ['sex', 'smoker', 'region']:
    print( "-------------------------\n The columns is "+ col,':' + "\n -------------------------- ")
    print(insurance[col].value_counts())
    


#                                                  ENCODING THE DATA 
Enoding the data as :
    Sex       = Female    : 0        Male      : 1
    Smoker    = No        : 0        Yes       : 1
    Region    = NorthEast : 0        NorthWest : 1        SouthEast : 2        SouthWest : 4
# In[21]:


# Importing Libraries for PROCESSING & ENCODING :

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


# In[22]:


# Encoding the object column (SEX, SMOKER, REGION) : 

for col in ['sex', 'smoker', 'region']:
    if (insurance[col].dtype == 'object'):
        le = preprocessing.LabelEncoder()
        le = le.fit(insurance[col])
        insurance[col] = le.transform(insurance[col])
        print('Encoding Completed for Column : ',col)

# Another method of Encoding :

 insurance['sex']=insurance['sex'].map({'female':0 , 'male':1 })
 insurance['smoker']=insurance['smoker'].map({'no':0 , 'yes':1 })
 insurance['region']=insurance['region'].map({'northeast':0 , ''northwest'':1 , 'southeast':2 , 'southwest':4 })
# In[23]:


# Printing tail value of the encoded data :

insurance.tail(10)


# In[25]:


# Saving the Encoded data in csv format :

insurance.to_csv('insurance_encoded.csv',index = False)


# In[26]:


# Getting the Co-relation by using heat map of all columns :

plt.figure(figsize=(8,8))

#setting the parameter values
cmap="tab20"
center=0

#setting the parameter values
annot= True

#plotting the heatmap
hm=sns.heatmap(insurance.corr(),cmap=cmap,annot=annot)

#displaying the plotted heatmap
plt.show()


#                                                MODEL PROCESSING

# In[27]:


# Splitting the Features and Target Variables :

x = insurance.drop(columns='charges', axis=1)
y = insurance['charges']


# In[28]:


print(x),print(y)


# In[29]:


# Splitting the data into Training data & Testing Data :

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


# In[30]:


# Printing Shape of X :

print(x.shape, x_train.shape, x_test.shape)


#                                                  APPLICATION OF MODELS

#                                                 LINEAR REGRESSION MODEL 
MODEL TRAINING
# In[31]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[32]:


# loading the Linear Regression model
reg = LinearRegression()


# In[33]:


reg.fit(x_train, y_train)

MODEL EVALUATION 
# In[34]:


# Prediction on Train data :

train_data_prediction =reg.predict(x_train)


# In[35]:


# R squared value of Train Data :

r2_train = metrics.r2_score(y_train, train_data_prediction)
print('R squared value of Train Data : ', r2_train)


# In[36]:


# Prediction on Test data :

test_data_prediction =reg.predict(x_test)


# In[37]:


# R squared value of Test Data :

r2_test = metrics.r2_score(y_test, test_data_prediction)
print('R squared value of Test Data: ', r2_test)

PREDICTIVE MODELLING SYSTEM
# In[38]:


# Inputting the values :
input_data= (18, 1, 10.000, 4,1,2)    

# Coverting into data to numpy array so as to avoid reshape error :
input_data_as_numpy_array = np.asarray(input_data)

# Reshapping the array :
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Interpreting the Predicted Result :
result = reg.predict(input_data_reshaped)
result


# In[39]:


# Checking the Accuracy :

accuracy = reg.score (x_test, y_test)
print(accuracy * 100, '%')  


#                                              RANDOM FOREST REGRESSION 

# In[40]:


# Knowing the X and y values :


# In[41]:


print(x)


# In[42]:


print(y)

MODEL TRAINING
# In[43]:


from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[44]:


# loading the Random Forest Regression model :

RFreg=RandomForestRegressor(n_estimators= 10 , random_state= 0)    # n_estimators represents number of nodes.


# In[45]:


RFreg.fit(x_train,y_train)

MODEL EVALUATION 
# In[46]:


# Prediction on Train data :

train_data_prediction =RFreg.predict(x_train)


# In[47]:


# Prediction on Test data :

test_data_prediction =RFreg.predict(x_test)


# In[48]:


# R squared value of Train Data :

r2_train = metrics.r2_score(y_train, train_data_prediction)
print('R squared value of Train Data: ', r2_train)


# In[49]:


# R squared value of Test Data :

r2_test = metrics.r2_score(y_test, test_data_prediction)
print('R squared value of Test Data : ', r2_test)

PREDICTIVE MODELLING SYSTEM    Sex       = Female    : 0        Male      : 1
    Smoker    = No        : 0        Yes       : 1
    Region    = NorthEast : 0        NorthWest : 1        SouthEast : 2        SouthWest : 4
# In[50]:


# Inputting the values :
input_data=  (18   , 1 , 33.770, 1 ,  0 , 2)        # age, sex , bmi, children, smoker, region
# Coverting into data to numpy array so as to avoid reshape error :
input_data_as_numpy_array = np.asarray(input_data)

# Reshapping the array :
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Interpreting the Predicted Result :
result = RFreg.predict(input_data_reshaped)
result


# In[51]:


# Checking the Accuracy :

accuracy = RFreg.score (x_test, y_test)
print(accuracy * 100, '%')

The Accuracy Score of Linear model is 74.45%.
The Accuracy Score Random Forest Regression is 82.60%.
The Random Forest Regression interprets the best accuracy.
Therefore we will proceed with the Random Forest Regression for futher prediction process.                                                     PICKLE FILE
# In[53]:


# Saving the Pickle file for futher process :

import pickle
pickle.dump(RFreg,open('insurance_claim.pkl','wb'))


# In[ ]:




