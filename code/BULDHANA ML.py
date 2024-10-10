#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv("BULDHANA.csv")
data


# In[2]:


from sklearn.ensemble import IsolationForest
model = IsolationForest(contamination=0.05)
outliers = model.fit_predict(data)
data = data[outliers!=-1]


# In[3]:


data.shape


# In[4]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def plot_correlation_heatmap(data, target_variable_name, figsize=(12, 9)):
    if target_variable_name not in data.columns:
        print(f"Error: '{target_variable_name}' not found in the dataset.")
        return
    target_variable = data[target_variable_name]
    correlation_with_target = data.corr()[target_variable_name].abs()
    N = min(55, len(data.columns))
    top_features = correlation_with_target.nlargest(N).index 
    data_subset = data[top_features]
    subset_correlation = data_subset.corr()
    print("\nCorrelation Matrix of Top Features:")
    print(subset_correlation)
    plt.figure(figsize=figsize)
    sns.heatmap(subset_correlation, annot=True, cmap='coolwarm', linewidths=2.5)
    plt.title("Correlation Heatmap of Top Features", fontsize=10)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()
plot_correlation_heatmap(data, 'Experimental weight', figsize=(25, 25))


# In[5]:


data.shape


# In[6]:


import statsmodels.api as sm
y = data['Experimental weight']
X = data.drop('Experimental weight', axis=1)
cols = list(X.columns)
pmax = 1
while len(cols) > 0:
    p = []
    X_1 = X[cols]
    model = sm.OLS(y, X_1).fit(drop_index=True)
    p = pd.Series(model.pvalues.values[1:], index=cols[1:])
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if pmax > 0.05:
        cols.remove(feature_with_p_max)
        X = X.drop(feature_with_p_max, axis=1) 
    else:
        break
data=data[cols]
data.insert(len(data.columns),'Experimental weight',y)


# In[7]:


data.shape


# In[8]:


data.columns


# In[9]:


X=data.drop(['Experimental weight'],axis=1)


# In[10]:


y=data['Experimental weight']


# In[11]:


import numpy as np
import pandas as pd
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
data = standard_scaler.fit_transform(data)
data = pd.DataFrame(data)


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
import numpy as np


# In[43]:


import seaborn as sns
import matplotlib.pyplot as plt
columns_to_plot = ['Weeds_2', 'Weeds_1', 'rf2fnaug22', 'rf2fnoct22', 'Sowing_Area',
       'NDVI_2fnoct22', 'rh_min_1fnaug22', 'rf1fnoct22', 'rh_min_1fnsep22',
       'RVIJul2fn', 'rf1fnaug22', 'rf2fnjul22', 'Crop condition_2',
       'rh_max_1fnsep22', 'Tmax_2fnjul22', 'Any_Damage_1', 'Weeds_0',
       'FAPAR_2fnJuly', 'Experimental weight']
sns.set(style="whitegrid")
plt.figure(figsize=(14, 8))
sns.histplot(data=data[columns_to_plot].melt(), x='value', hue='variable', multiple='stack', bins=20, kde=False, palette="viridis")
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()


# In[44]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
sns.boxplot(data=data)
plt.title("BackView")
plt.xticks(rotation=70)
plt.show()


# # linear

# In[15]:


from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
model_lr.fit(X_train,y_train)
y_pred=model_lr.predict(X_test)
from sklearn.metrics import r2_score
print("accuracy",r2_score(y_test, y_pred))
print("mse",mse(y_test, y_pred))
print("mae",mae(y_test, y_pred))


# In[16]:


import seaborn as sns
sns.regplot(y_test,y_pred)


# In[17]:


from sklearn.model_selection import cross_val_score,cross_val_predict
x_combined = np.concatenate((X_train, X_test), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)
model_lr = LinearRegression()
cv_scores = cross_val_score(model_lr, x_combined, y_combined, cv=7, scoring='r2')
best_fold_index = np.argmax(cv_scores)
y_pred_cv = cross_val_predict(model_lr, x_combined, y_combined, cv=7)
y_pred_best_fold = y_pred_cv[y_combined == y_combined[best_fold_index]]
print("Cross-Validation Scores (R2):", cv_scores)


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_combined, y=y_pred_cv, label="Cross-Validation Predictions", alpha=0.7)
best_fold_mask = (y_combined == y_combined[best_fold_index])
sns.scatterplot(x=y_combined[best_fold_mask], y=y_pred_best_fold,
                label=f"Best Fold (R2: {cv_scores[best_fold_index]:.4f})", color='yellow', s=50)
plt.plot([min(y_combined), max(y_combined)], [min(y_combined), max(y_combined)], linestyle='--', color='black')
plt.title("Linear Regression")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()


# # random forest

# In[19]:


from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor()
model_rf.fit(X_train,y_train)
y_pred=model_rf.predict(X_test)
from sklearn.metrics import r2_score
print("accuracy",r2_score(y_test, y_pred))
print("mse",mse(y_test, y_pred))
print("mae",mae(y_test, y_pred))


# In[20]:


sns.regplot(y_test,y_pred)


# In[21]:


from sklearn.model_selection import cross_val_score,cross_val_predict
x_combined = np.concatenate((X_train, X_test), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)
model_rf = RandomForestRegressor()
cv_scores = cross_val_score(model_rf, x_combined, y_combined, cv=7, scoring='r2')
best_fold_index = np.argmax(cv_scores)
y_pred_cv = cross_val_predict(model_rf, x_combined, y_combined, cv=7)
y_pred_best_fold = y_pred_cv[y_combined == y_combined[best_fold_index]]
print("Cross-Validation Scores (R2):", cv_scores)


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_combined, y=y_pred_cv, label="Cross-Validation Predictions", alpha=0.7)
best_fold_mask = (y_combined == y_combined[best_fold_index])
sns.scatterplot(x=y_combined[best_fold_mask], y=y_pred_best_fold,
                label=f"Best Fold (R2: {cv_scores[best_fold_index]:.4f})", color='yellow', s=50)
plt.plot([min(y_combined), max(y_combined)], [min(y_combined), max(y_combined)], linestyle='--', color='black')
plt.title("RandomForest")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()


# # decision tree

# In[23]:


from sklearn.tree import DecisionTreeRegressor
model_dt = DecisionTreeRegressor()
model_dt.fit(X_train,y_train)
y_pred=model_dt.predict(X_test)
from sklearn.metrics import r2_score
print("accuracy",r2_score(y_test, y_pred))
print("mse",mse(y_test, y_pred))
print("mae",mae(y_test, y_pred))


# In[24]:


sns.regplot(y_test,y_pred)


# In[25]:


from sklearn.model_selection import cross_val_score,cross_val_predict
x_combined = np.concatenate((X_train, X_test), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)
model_dt = DecisionTreeRegressor()
cv_scores = cross_val_score(model_dt, x_combined, y_combined, cv=7, scoring='r2')
best_fold_index = np.argmax(cv_scores)
y_pred_cv = cross_val_predict(model_dt, x_combined, y_combined, cv=7)
y_pred_best_fold = y_pred_cv[y_combined == y_combined[best_fold_index]]
print("Cross-Validation Scores (R2):", cv_scores)


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_combined, y=y_pred_cv, label="Cross-Validation Predictions", alpha=0.7)
best_fold_mask = (y_combined == y_combined[best_fold_index])
sns.scatterplot(x=y_combined[best_fold_mask], y=y_pred_best_fold,
                label=f"Best Fold (R2: {cv_scores[best_fold_index]:.4f})", color='yellow', s=50)
plt.plot([min(y_combined), max(y_combined)], [min(y_combined), max(y_combined)], linestyle='--', color='black')
plt.title("DecisionTree")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()


# # xgb

# In[27]:


from xgboost import XGBRegressor
model_xg = XGBRegressor()
model_xg.fit(X_train,y_train)
y_pred=model_xg.predict(X_test)
from sklearn.metrics import r2_score
print("accuracy",r2_score(y_test, y_pred))
print("mse",mse(y_test, y_pred))
print("mae",mae(y_test, y_pred))


# In[28]:


sns.regplot(y_test,y_pred)


# In[29]:


from sklearn.model_selection import cross_val_score,cross_val_predict
x_combined = np.concatenate((X_train, X_test), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)
model_xg = XGBRegressor()
cv_scores = cross_val_score(model_xg, x_combined, y_combined, cv=7, scoring='r2')
best_fold_index = np.argmax(cv_scores)
y_pred_cv = cross_val_predict(model_xg, x_combined, y_combined, cv=7)
y_pred_best_fold = y_pred_cv[y_combined == y_combined[best_fold_index]]
print("Cross-Validation Scores (R2):", cv_scores)


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_combined, y=y_pred_cv, label="Cross-Validation Predictions", alpha=0.7)
best_fold_mask = (y_combined == y_combined[best_fold_index])
sns.scatterplot(x=y_combined[best_fold_mask], y=y_pred_best_fold,
                label=f"Best Fold (R2: {cv_scores[best_fold_index]:.4f})", color='yellow', s=50)
plt.plot([min(y_combined), max(y_combined)], [min(y_combined), max(y_combined)], linestyle='--', color='black')
plt.title("XGB")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()


# # svr

# In[31]:


from sklearn.svm import SVR
model_svr = SVR()
model_svr.fit(X_train,y_train)
y_pred=model_svr.predict(X_test)
from sklearn.metrics import r2_score
print("accuracy",r2_score(y_test, y_pred))
print("mse",mse(y_test, y_pred))
print("mae",mae(y_test, y_pred))


# In[32]:


sns.regplot(y_test,y_pred)


# In[33]:


from sklearn.model_selection import cross_val_score,cross_val_predict
x_combined = np.concatenate((X_train, X_test), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)
model_svr = SVR()
cv_scores = cross_val_score(model_svr, x_combined, y_combined, cv=7, scoring='r2')
best_fold_index = np.argmax(cv_scores)
y_pred_cv = cross_val_predict(model_svr, x_combined, y_combined, cv=7)
y_pred_best_fold = y_pred_cv[y_combined == y_combined[best_fold_index]]
print("Cross-Validation Scores (R2):", cv_scores)


# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_combined, y=y_pred_cv, label="Cross-Validation Predictions", alpha=0.7)
best_fold_mask = (y_combined == y_combined[best_fold_index])
sns.scatterplot(x=y_combined[best_fold_mask], y=y_pred_best_fold,
                label=f"Best Fold (R2: {cv_scores[best_fold_index]:.4f})", color='yellow', s=50)
plt.plot([min(y_combined), max(y_combined)], [min(y_combined), max(y_combined)], linestyle='--', color='black')
plt.title("SVR")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()


# In[35]:


pip install shap


# In[36]:


import shap
from sklearn.ensemble import RandomForestRegressor


# In[37]:


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[38]:


explainer = shap.TreeExplainer(rf_model)


# In[39]:


shap_values = explainer.shap_values(X_train)


# In[40]:


shap.summary_plot(shap_values, X_train, feature_names=X_train.columns)


# In[41]:


sample_idx = 0
shap.force_plot(explainer.expected_value, shap_values[sample_idx, :], X_train.iloc[sample_idx], feature_names=X_train.columns)


# In[42]:


most_impactful_features = X_train.columns[np.abs(shap_values).mean(axis=0).argsort()[::-1]]
print("Most impactful features:", most_impactful_features)


# In[ ]:




