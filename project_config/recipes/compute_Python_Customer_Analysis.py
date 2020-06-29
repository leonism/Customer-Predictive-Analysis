# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

from plotly.offline import init_notebook_mode,iplot
import plotly.graph_objects as go
init_notebook_mode(connected=True)


# Read recipe inputs
customer_Analysis = dataiku.Dataset("Customer_Analysis")
# customer_Analysis_df = customer_Analysis.get_dataframe()
df = customer_Analysis.get_dataframe()

# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

python_Customer_Analysis_df = df # For this sample code, simply copy input to output


# Write recipe outputs
python_Customer_Analysis = dataiku.Dataset("Python_Customer_Analysis")
python_Customer_Analysis.write_with_schema(python_Customer_Analysis_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # EDA (Exploratory Data Analysis)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.describe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
lab = df["Response"].value_counts().keys().tolist() #Labels
val = df["Response"].value_counts().values.tolist() #Values
trace = go.Pie(labels=lab,
                values=val,
                marker=dict(colors=['blue']),
                hoverinfo="value")

layout = go.Layout(title="Response Distribution")

fig = go.Figure(data = trace,layout = layout)

iplot(fig)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import plotly.express as px

fig = px.histogram(df, x="Sales Channel", color='Response')
fig.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
fig = px.histogram(df, x="Renew Offer Type", color='Response')
fig.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
fig = px.histogram(df, x="Education", color='Response')
fig.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
fig = px.histogram(df, x="EmploymentStatus", color='Response')
fig.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
fig = px.histogram(df, x="Vehicle Class", color='Response')
fig.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
fig = px.histogram(df, x="Policy", color='Response')
fig.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
fig = px.violin(df, y="Total Claim Amount", x="Response",
                box=True, points="all", # hover_data=df.columns,
                title = 'Response Rate by Total Claim Amount')
fig.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
fig = px.violin(df, y="Income", x="Response",
                box=True, points="all", # hover_data=df.columns,
                title = 'Response Rate by Income')
fig.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
fig = px.scatter_3d(df, x="Income", y="Monthly Premium Auto",
                    z="Customer Lifetime Value",
                    color="Response", size="Income",
                    color_discrete_map = {"Income": "blue",
                                          "Monthly Premium Auto": "green",
                                          "Customer Lifetime Value":"red"})
fig.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Regression Analysis with Continuous Variables

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# make response a numerical value
df.Response = df.Response.apply(lambda X : 0 if X == 'No' else 1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import statsmodels.api as sm # ok

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# get only the numerical columns from the dataset
cont_df = df.select_dtypes(include=['int64','float'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
cont_df.head(3)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# get their number of unique distinct value
cont_df.nunique()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
cont_reg = sm.Logit(cont_df['Response'],
                             cont_df.drop('Response', axis = 1))
cont_reg.fit().summary()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.heatmap(cont_df.corr(), annot = True)
plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Regression Analysis with Categorical Variables

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
categorical_df = df.select_dtypes(include='object')
cat_df = categorical_df.drop(['Customer','Effective To Date'], axis = 1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
cat_df.nunique()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
cols=cat_df.columns.tolist()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
for col in cat_df[cols]:
    cat_df[col] = lb.fit_transform(cat_df[col])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.preprocessing import LabelEncoder
for col in cat_df[cols]:
    cat_df[col] = LabelEncoder().fit_transform(cat_df[col])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
cat_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
categorical_train = sm.Logit(cont_df.Response, cat_df)
categorical_train.fit().summary()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Regression Analysis with both Continuous and Categorical Variables

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
cont_df.reset_index(drop = True, inplace=True)
cat_df.reset_index(drop = True, inplace=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
combined_df = pd.concat([cont_df,cat_df], axis = 1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
combined_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
combined_train = sm.Logit(combined_df.Response, combined_df.drop(['Response'], axis = 1))
combined_train.fit().summary()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Regression Analysis with excluding Non-significant variables

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
significant_cols = ['Customer Lifetime Value','Income','Monthly Premium Auto','Months Since Last Claim',
                    'Months Since Policy Inception','Number of Policies','Total Claim Amount','Marital Status',
                    'Renew Offer Type','Sales Channel','Vehicle Size']
trainData = sm.Logit(combined_df.Response, combined_df[significant_cols])
trainData.fit().summary()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Classification

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
y = combined_df.Response
X = combined_df.drop('Response', axis = 1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.model_selection import train_test_split, cross_validate

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # SVC Classification

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC

svc = SVC(gamma='auto')
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)

print(confusion_matrix(svc_pred,y_test))
print('Accuracy_score:',accuracy_score(svc_pred, y_test))
print(classification_report(svc_pred, y_test))

cross_val_score_svc = cross_validate(svc, X_train, y_train,cv = 5,return_train_score=True)
print('Cross validation Train_score',cross_val_score_svc['train_score'].mean())
print('Cross validation Test_score',cross_val_score_svc['test_score'].mean())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Random Forest Classification

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print(confusion_matrix(rfc_pred,y_test))
print('Accuracy score:',accuracy_score(rfc_pred, y_test))
print(classification_report(rfc_pred, y_test))

cross_val_score_rfc = cross_validate(rfc, X_train, y_train,cv = 5,return_train_score=True)

print('Cross validation train_score',cross_val_score_rfc['train_score'].mean())
print('Cross validation test_score',cross_val_score_rfc['test_score'].mean())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Feature Importance

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
feature_imp = rfc.feature_importances_.round(3)
ser_rank = pd.Series(feature_imp,
                     index=X.columns).sort_values(ascending = False)

plt.figure(figsize=(12,7))
sns.barplot(x= ser_rank.values, y = ser_rank.index, palette='deep')
plt.xlabel('relative importance')
plt.show()