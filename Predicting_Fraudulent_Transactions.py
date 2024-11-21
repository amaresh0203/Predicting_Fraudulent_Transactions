#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[2]:


data = pd.read_csv('Fraud.csv')


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)


# In[7]:


# Check for outliers
outlier_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
for col in outlier_cols:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]


# In[8]:


# Address multicollinearity
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data


# In[9]:


# Select relevant columns for VIF calculation
selected_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Calculate VIF
vif_result = calculate_vif(data[selected_columns])
print("VIF Result:\n", vif_result)


# In[10]:


# Drop columns with high VIF (VIF > 5 is often considered high)
high_vif_columns = vif_result[vif_result['VIF'] > 5]['Variable'].tolist()
data = data.drop(high_vif_columns, axis=1)


# In[11]:


data.head()


# In[13]:


data.info()


# In[14]:


data.isFraud.value_counts()


# In[15]:


data.isFlaggedFraud.value_counts()


# In[16]:


data=data.drop(['nameOrig','nameDest'],axis=1)


# In[17]:


data.head()


# In[18]:


data = pd.get_dummies(data, columns=['type'], drop_first=True)


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[21]:


from sklearn.model_selection import train_test_split
x = data.drop(['isFraud'], axis=1)
y = data['isFraud']


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[23]:


train_data=x_train.join(y_train)


# In[24]:


train_data


# In[25]:


from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.svm import SVR,SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score,mean_squared_error


# In[34]:


regressors = [['Linear Regression:', LinearRegression()],
              ['Decision Tree Regression:', DecisionTreeRegressor()],
              ['Random Forest Regression:', RandomForestRegressor()],
              ['Extra Tree Regression:', ExtraTreesRegressor()]]
        

reg_pred = []
print('Results...\n')

for name, model in regressors:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    rms = np.sqrt(mean_squared_error(y_test, predictions))
    reg_pred.append(rms)
    print(name, rms)


# In[35]:


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")


# In[50]:


# Convert regression problem to classification
threshold = 0.5  
y_pred_class = np.where(predictions >= threshold, 1, 0)
y_test_class = np.where(y_test >= threshold, 1, 0)

regressors = [
    ['Linear Regression:', LinearRegression()],
    ['Decision Tree Regression:', DecisionTreeRegressor()],
    ['Random Forest Regression:', RandomForestRegressor()],
    ['Extra Tree Regression:', ExtraTreesRegressor()]
]

reg_acc = []

print('Results...\n')

for name, model in regressors:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    
    # Convert regression predictions to classification
    y_pred_class = np.where(predictions >= threshold, 1, 0)
    y_test_class = np.where(y_test >= threshold, 1, 0)
    
    accuracy = accuracy_score(y_test_class, y_pred_class)
    reg_acc.append(accuracy)
    print(name, accuracy)


# In[37]:


feature_importances = pd.Series(model.feature_importances_, index=x.columns)
sorted_importances = feature_importances.sort_values(ascending=False)
print(sorted_importances)


# In[39]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[41]:


# Visualize Feature Importance
feature_importances = pd.Series(model.feature_importances_, index=x.columns)
sorted_importances = feature_importances.sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_importances, y=sorted_importances.index, palette='viridis')
plt.xlabel('Feature Importance')
plt.title('Feature Importance Plot')
plt.show()


# In[47]:


y_ax=['LinearRegression' ,'DecisionTreeRegression', 'RandomForestRegression','ExtraTreeRegression']
x_ax=reg_pred


# In[51]:


sns.barplot(x=x_ax,y=y_ax,linewidth=1.5,edgecolor="0.1")
plt.xlabel('RMSE')
plt.title('Best Regressor Plot')
plt.show()


# In[ ]:




