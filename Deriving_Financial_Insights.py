#!/usr/bin/env python
# coding: utf-8

# **Your challenge can be found toward the end of this notebook. The code below will be needed in order to begin the challenge. Read through and execute all necessary portions of this code to complete the tasks for this challenge.**

# ##### Import the necessary packages


import numpy as np #numerical computation
import pandas as pd #data wrangling
import matplotlib.pyplot as plt #plotting package

#Next line helps with rendering plots
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl #add'l plotting functionality
mpl.rcParams['figure.dpi'] = 400 #high res figures
import graphviz #to visualize decision trees


# ##### Cleaning the Dataset

# In[5]:


df_orig = pd.read_excel('default_of_credit_card_clients.xls')
df_orig.head()


# In[6]:


df_zero_mask = df_orig == 0


# In[7]:


feature_zero_mask = df_zero_mask.iloc[:,1:].all(axis=1)
feature_zero_mask


# In[8]:


sum(feature_zero_mask)


# Remove all the rows with all zero features and response, confirm this that gets rid of the duplicate IDs.

# In[9]:


df_clean = df_orig.loc[~feature_zero_mask,:].copy()


# In[10]:


df_clean.shape


# In[11]:


df_clean['ID'].nunique()


# Clean up the `EDUCATION` and `MARRIAGE` features as in Chapter 1

# In[12]:


df_clean['EDUCATION'].value_counts()


# "Education (1 = graduate school; 2 = university; 3 = high school; 4 = others)"

# Assign unknown categories to other.

# In[13]:


df_clean['EDUCATION'].replace(to_replace=[0, 5, 6], value=4, inplace=True)


# In[14]:


df_clean['EDUCATION'].value_counts()


# Examine and clean marriage feature as well:

# In[15]:


df_clean['MARRIAGE'].value_counts()


# In[16]:


#Should only be (1 = married; 2 = single; 3 = others).
df_clean['MARRIAGE'].replace(to_replace=0, value=3, inplace=True)


# In[17]:


df_clean['MARRIAGE'].value_counts()


# Now instead of removing rows with `PAY_1` = 'Not available', as done in Chapter 1, here select these out for addition to training and testing splits.

# In[18]:


df_clean['PAY_1'].value_counts()


# In[19]:


missing_pay_1_mask = df_clean['PAY_1'] == 'Not available'


# In[20]:


sum(missing_pay_1_mask)


# In[21]:


df_missing_pay_1 = df_clean.loc[missing_pay_1_mask,:].copy()


# In[22]:


df_missing_pay_1.shape


# In[23]:


df_missing_pay_1['PAY_1'].head(3)


# In[24]:


df_missing_pay_1['PAY_1'].value_counts()


# In[25]:


df_missing_pay_1.columns


# Load cleaned data

# In[26]:


df = pd.read_csv('cleaned_data.csv')
df.head()


# In[27]:


df.columns


# In[28]:


features_response = df.columns.tolist()
features_response


# In[29]:


items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university']


# In[30]:


features_response = [item for item in features_response if item not in items_to_remove]
features_response


# ##### Mode and Random Imputation of `PAY_1`

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(df[features_response[:-1]].values, df['default payment next month'].values,test_size=0.2, random_state=24)


# In[33]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[34]:


df_missing_pay_1.shape


# In[35]:


features_response[4]


# In[36]:


np.median(X_train[:,4])


# In[37]:


np.random.seed(seed=1)
fill_values = [0, np.random.choice(X_train[:,4], size=(3021,), replace=True)]
fill_values


# In[38]:


fill_strategy = ['mode', 'random']
fill_strategy 


# In[39]:


fill_values[-1]


# In[40]:


fig, axs = plt.subplots(1,2, figsize=(8,3))
bin_edges = np.arange(-2,9)
axs[0].hist(X_train[:,4], bins=bin_edges, align='left')
axs[0].set_xticks(bin_edges)
axs[0].set_title('Non-missing values of PAY_1')
axs[1].hist(fill_values[-1], bins=bin_edges, align='left')
axs[1].set_xticks(bin_edges)
axs[1].set_title('Random selection for imputation')
plt.tight_layout()


# To do cross-validation on the training set, now we need to shuffle since all the samples with missing `PAY_1` were concatenated on to the end.

# In[41]:


from sklearn.model_selection import KFold


# In[42]:


k_folds = KFold(n_splits=4, shuffle=True, random_state=1)
k_folds


# Don't need to do a grid search, so we can use `cross_validate`

# In[43]:


from sklearn.model_selection import cross_validate


# For the estimator, set the optimal hyperparameters determined in previous chapter.

# In[44]:


from sklearn.ensemble import RandomForestClassifier


# In[45]:


rf = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=9,
min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
random_state=4, verbose=1, warm_start=False, class_weight=None)

rf


# In[46]:


for counter in range(len(fill_values)):
    #Copy the data frame with missing PAY_1 and assign imputed values
    df_fill_pay_1_filled = df_missing_pay_1.copy()
    df_fill_pay_1_filled['PAY_1'] = fill_values[counter]
    
    #Split imputed data in to training and testing, using the same 80/20 split we have used for the data with non-missing PAY_1
    X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test =     train_test_split(
        df_fill_pay_1_filled[features_response[:-1]].values,
        df_fill_pay_1_filled['default payment next month'].values,
    test_size=0.2, random_state=24)
    
    #Concatenate the imputed data with the array of non-missing data
    X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
    y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)
    
    #Use the KFolds splitter and the random forest model to get 4-fold cross-validation scores for both imputation methods
    imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise-deprecating')
    
    test_score = imputation_compare_cv['test_score']
    print(fill_strategy[counter] + ' imputation: ' +
          'mean testing score ' + str(np.mean(test_score)) +
          ', std ' + str(np.std(test_score)))


# ##### A Predictive Model for `PAY_1`

# In[47]:


pay_1_df = df.copy()
pay_1_df.head()


# In[48]:


features_for_imputation = pay_1_df.columns.tolist()
features_for_imputation


# In[49]:


items_to_remove_2 = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university', 'default payment next month', 'PAY_1']


# In[50]:


features_for_imputation = [item for item in features_for_imputation if item not in items_to_remove_2]
features_for_imputation


# ##### Building a Multiclass Classification Model for Imputation

# In[51]:


X_impute_train, X_impute_test, y_impute_train, y_impute_test = train_test_split(
    pay_1_df[features_for_imputation].values,
    pay_1_df['PAY_1'].values,
test_size=0.2, random_state=24)


# In[52]:


rf_impute_params = {'max_depth':[3, 6, 9, 12],
             'n_estimators':[10, 50, 100, 200]}

rf_impute_params


# In[53]:


from sklearn.model_selection import GridSearchCV


# Need to use accuracy here as ROC AUC is not supported for multiclass. Need to use multiclass and not regression because need to limit to integer values of `PAY_1`.

# In[54]:


cv_rf_impute = GridSearchCV(rf, param_grid=rf_impute_params, scoring='accuracy',
                            n_jobs=-1, iid=False, refit=True,
                            cv=4, verbose=2, error_score=np.nan, return_train_score=True)

cv_rf_impute


# In[55]:


cv_rf_impute.fit(X_impute_train, y_impute_train)


# In[56]:


impute_df = pd.DataFrame(cv_rf_impute.cv_results_)
impute_df


# In[60]:


cv_rf_impute.best_params_


# In[61]:


cv_rf_impute.best_score_


# In[62]:


pay_1_value_counts = pay_1_df['PAY_1'].value_counts().sort_index()


# In[63]:


pay_1_value_counts


# In[64]:


pay_1_value_counts/pay_1_value_counts.sum()


# In[65]:


y_impute_predict = cv_rf_impute.predict(X_impute_test)


# In[66]:


from sklearn import metrics


# In[67]:


metrics.accuracy_score(y_impute_test, y_impute_predict)


# In[68]:


fig, axs = plt.subplots(1,2, figsize=(8,3))
axs[0].hist(y_impute_test, bins=bin_edges, align='left')
axs[0].set_xticks(bin_edges)
axs[0].set_title('Non-missing values of PAY_1')
axs[1].hist(y_impute_predict, bins=bin_edges, align='left')
axs[1].set_xticks(bin_edges)
axs[1].set_title('Model-based imputation')
plt.tight_layout()


# In[69]:


X_impute_all = pay_1_df[features_for_imputation].values
y_impute_all = pay_1_df['PAY_1'].values


# In[70]:


rf_impute = RandomForestClassifier(n_estimators=100, max_depth=12)


# In[71]:


rf_impute


# In[72]:


rf_impute.fit(X_impute_all, y_impute_all)


# ##### Using the Imputation Model and Comparing it to Other Methods

# In[73]:


df_fill_pay_1_model = df_missing_pay_1.copy()


# In[74]:


df_fill_pay_1_model['PAY_1'].head()


# In[75]:


df_fill_pay_1_model['PAY_1'] = rf_impute.predict(df_fill_pay_1_model[features_for_imputation].values)


# In[76]:


df_fill_pay_1_model['PAY_1'].head()


# In[77]:


df_fill_pay_1_model['PAY_1'].value_counts().sort_index()


# In[78]:


X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = train_test_split(
    df_fill_pay_1_model[features_response[:-1]].values,
    df_fill_pay_1_model['default payment next month'].values,
test_size=0.2, random_state=24)


# In[79]:


print(X_fill_pay_1_train.shape)
print(X_fill_pay_1_test.shape)
print(y_fill_pay_1_train.shape)
print(y_fill_pay_1_test.shape)


# In[80]:


X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)


# In[81]:


print(X_train_all.shape)
print(y_train_all.shape)


# In[82]:


rf


# In[83]:


imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise-deprecating')


# In[84]:


imputation_compare_cv['test_score']


# In[85]:


np.mean(imputation_compare_cv['test_score'])


# In[86]:


np.std(imputation_compare_cv['test_score'])


# Reassign values using mode imputation

# In[87]:


df_fill_pay_1_model['PAY_1'] = np.zeros_like(df_fill_pay_1_model['PAY_1'].values)


# In[88]:


df_fill_pay_1_model['PAY_1'].unique()


# In[89]:


X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = train_test_split(
    df_fill_pay_1_model[features_response[:-1]].values,
    df_fill_pay_1_model['default payment next month'].values,
test_size=0.2, random_state=24)


# In[90]:


X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
X_test_all = np.concatenate((X_test, X_fill_pay_1_test), axis=0)
y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)
y_test_all = np.concatenate((y_test, y_fill_pay_1_test), axis=0)


# In[91]:


print(X_train_all.shape)
print(X_test_all.shape)
print(y_train_all.shape)
print(y_test_all.shape)


# In[92]:


imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise-deprecating')


# In[93]:


np.mean(imputation_compare_cv['test_score'])


# ##### Confirming Model Performance on the Unseen Test Set

# In[94]:


rf.fit(X_train_all, y_train_all)


# In[95]:


y_test_all_predict_proba = rf.predict_proba(X_test_all)


# In[96]:


from sklearn.metrics import roc_auc_score


# In[97]:


roc_auc_score(y_test_all, y_test_all_predict_proba[:,1])


# ##### Characterizing Costs and Savings

# In[98]:


thresholds = np.linspace(0, 1, 101)
thresholds


# Use mean bill amount to estimate savings per prevented default

# In[99]:


df[features_response[:-1]].columns[5]


# In[100]:


savings_per_default = np.mean(X_test_all[:, 5])
savings_per_default


# In[101]:


cost_per_counseling = 7500


# In[102]:


effectiveness = 0.70


# In[103]:


n_pos_pred = np.empty_like(thresholds)
cost_of_all_counselings = np.empty_like(thresholds)
n_true_pos = np.empty_like(thresholds)
savings_of_all_counselings = np.empty_like(thresholds)


# In[104]:


counter = 0
for threshold in thresholds:
    pos_pred = y_test_all_predict_proba[:,1]>threshold
    n_pos_pred[counter] = sum(pos_pred)
    cost_of_all_counselings[counter] = n_pos_pred[counter] * cost_per_counseling
    true_pos = pos_pred & y_test_all.astype(bool)
    n_true_pos[counter] = sum(true_pos)
    savings_of_all_counselings[counter] = n_true_pos[counter] * savings_per_default * effectiveness
    
    counter += 1


# In[105]:


net_savings = savings_of_all_counselings - cost_of_all_counselings


# In[106]:


plt.plot(thresholds, cost_of_all_counselings)
plt.xlabel('Threshold')
plt.ylabel('Cost of all counselings')
plt.grid()
plt.show()


# In[107]:


plt.plot(thresholds, savings_of_all_counselings)
plt.xlabel('Threshold')
plt.ylabel('Saving of all counselings')
plt.grid()
plt.show()


# In[108]:


# Net Savings
mpl.rcParams['figure.dpi'] = 400
plt.plot(thresholds, net_savings)
plt.xlabel('Threshold')
plt.ylabel('Net savings (NT$)')
plt.xticks(np.linspace(0,1,11))
plt.grid(True)


# In[109]:


max_savings_ix = np.argmax(net_savings)
max_savings_ix


# What is the threshold at which maximum savings is achieved?

# In[110]:


thresholds[max_savings_ix]


# What is the maximum possible savings?

# In[111]:


net_savings[max_savings_ix]


# ## Challenge: Deriving Financial Insights

# In[112]:


# This will autosave your notebook every ten seconds
get_ipython().run_line_magic('autosave', '10')


# **Using the testing set, calculate the cost of all defaults if there were no counseling program and output your result.**

# In[113]:


savings_per_default = np.mean(X_test_all[:, 5])
savings_per_default


# In[114]:


cost_of_defaults = sum(y_test_all) * savings_per_default
cost_of_defaults


# **This the Cost of all defaults assuming with no counseling program**

# _______________________________________________________________________________________________
# **Next, calculate by what percent can the cost of defaults be decreased by the counseling program and output you result.**

# In[115]:


net_savings[max_savings_ix]/cost_of_defaults


# **Results indicate that we can decrease the cost of defaults by 23% using a counseling program, guided by predictive modeling.**

# _______________________________________________________________________________________________
# **Then, calculate the net savings per account at the optimal threshold and output your result.**

# In[116]:


savings_of_all_counselings= n_true_pos * savings_per_default * effectiveness
savings_of_all_counselings


# In[117]:


net_savings = savings_of_all_counselings - cost_of_all_counselings
net_savings


# In[118]:


net_savings[max_savings_ix]/len(y_test_all)


# **This is amount of savings the company could create with the counseling program, to as many accounts as they serve.**

# _______________________________________________________________________________________________
# **Now, plot the net savings per account against the cost of counseling per account for each threshold.**

# In[119]:


import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400
plt.plot(thresholds, net_savings)
plt.xlabel('Threshold')
plt.ylabel('Net savings (NT$)')
plt.xticks(np.linspace(0,1,11))
plt.grid(True)
plt.show()


# **The above graph indicates how much money the client have to budget for the counseling program in a given month, to achieve a given amount of savings. It looks like the greatest benefit can be created by budgeting up to NT 2000 dollars per account. After this, net savings are relatively flat, and then decline.**

# _______________________________________________________________________________________________
# **Next, plot the fraction of accounts predicted as positive (this is called the "flag rate") at each threshold.**

# In[120]:


plt.plot(thresholds, n_pos_pred/len(y_test_all), color='red')
plt.ylabel('Flag rate')
plt.xlabel('Threshold')
plt.grid(True)


# **The plot shows the fraction of people who will be predicted to default. It appears that at the optimal threshold of 0.2, only about 30% of accounts will be flagged for counseling.**

# _______________________________________________________________________________________________
# **Next, plot a precision-recall curve for the testing data.**

# In[121]:


plt.plot(n_true_pos/sum(y_test_all), np.divide(n_true_pos, n_pos_pred), color='green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)


# **To start getting a true positive rate (recall) much above 0, we need to accept a precision of about 0.75 or lower. Therefore, it appears that there is chance for improvement in our model.**

# _______________________________________________________________________________________________
# **Finally, plot precision and recall separately on the y-axis against threshold on the x-axis.**

# In[122]:


plt.plot(thresholds, n_true_pos/sum(y_test_all), label='Recall')
plt.plot(thresholds, np.divide(n_true_pos, n_pos_pred),label='Precision')

plt.xlabel('Threshold')
plt.ylabel('Precision & Recall')
plt.grid()
plt.legend(loc='best')


# **The above plot shows that the optimal threshold is 0.2. Here the optimal threshold also depends on the financial analysis of costs and savings, we can see here that the steepest part of the initial increase in precision, which constitutes the correctness of positive predictions and a measure of how cost-effective the model-guided counseling can be done up to a threshold of about 0.2.**

# In[124]:


import pickle
pickle.dump(rf, open("loan.pkl", "wb"))

