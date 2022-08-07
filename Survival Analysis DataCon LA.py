# import libraries
import openpyxl
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
import matplotlib as plt
from matplotlib import pyplot as plt
from lifelines.utils import median_survival_times
import plotly.io as pio
pio.renderers.default = "browser"

def load_data():
    """Loads data into the dataframe"""
    df = pd.read_excel(r'../DataConGit/CustomerBaseDataConLA.xlsx',
                       sheet_name='Sheet1')
    return df

def find_missing_data(df):
    """Returns columns with count of missing values as % of total rows"""
    null_data = df.isnull().sum()[df.isnull().sum() > 0]
    data_dict = {'count': null_data.values, '%': np.round(null_data.values * 100 / df.shape[0], 2) }
    df_null = pd.DataFrame(data=data_dict, index=null_data.index)
    df_null.sort_values(by='count', ascending=False, inplace=True)
    print("~~~Count of missing values in each column as % of total rows~~~")
    return df_null

def clean_column_names(df):
    """Concatenates words in column names resulting in a single string
    Churn Date --> ChurnDate"""
    df.columns = df.columns.str.replace(' ', '')

def create_dummy_df(df, cat_cols):
    for col in cat_cols:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True)], axis=1)
        except:
            continue
    return df

#load and clean data
df = load_data()
clean_column_names(df)
find_missing_data(df)

#KAPLAN-MEIER estimator
'''Kaplan-Meyer is a popular method to analyze ‘time-to-event’ data and a standard estimator of the survival function, 
which tells us the average survival probability of the entire population at each time period. 
It takes into account “censored” or incomplete data and returns the cumulative probability of surviving past a certain point in time.'''

T = df['ContractAge']
E = df['Churned']
kmf = KaplanMeierFitter().fit(T, E, label="all_types")
ax = kmf.survival_function_.plot()
plt.title('Kaplan Meier: Survival function of ALL subscribers');
plt.axhline(0.5, color='red', alpha=0.75, linestyle=':')
plt.xlabel('Time since contract start')
plt.ylabel('Probability still a customer')
ax.set_ylim([0.0, 1.0])

print(f'median survival years:{round(kmf.median_survival_time_,2)}') #The median time of contract, which defines the point in time where on average 50% of the population has churned

median_ci = median_survival_times(kmf.confidence_interval_)
print(round(median_ci,2)) #The lower and upper confidence intervals for the survival function, represents when 50% of the population has churned

#segmenting by customer type
ax = plt.subplot(111)
typ = (df["CustomerType"] == "A")
kmf.fit(T[typ], event_observed=E[typ], label="Cohort A")
kmf.plot_survival_function(ax=ax)
kmf.fit(T[~typ], event_observed=E[~typ], label="vs. Rest")
kmf.plot_survival_function(ax=ax)
plt.axhline(0.5, color='red', alpha=0.75, linestyle=':')
plt.title("Survival function by Customer Type");
plt.xlabel('Time since contract start')
plt.ylabel('Probability still a customer')
ax.set_ylim([0.0, 1.0])
#kmf.survival_function_.to_excel("misc_output.xlsx")

print(f'median survival years:{round(kmf.median_survival_time_,2)}')
median_ci = median_survival_times(kmf.confidence_interval_)
print(round(median_ci,2))

customer_cohorts = df['CustomerType'].dropna().unique()
for i, x in enumerate(customer_cohorts):
    ax = plt.subplot(2, 3, i + 1)
    ix = df['CustomerType'] == x
    kmf.fit(T[ix], E[ix], label=x)
    kmf.plot_survival_function(ax=ax, legend=False)
    plt.title(x)
    plt.axhline(0.5, color='red', alpha=0.75, linestyle=':')
    plt.xlim(0, 30)
    plt.xlabel('Time since contract start')
    plt.ylabel('Probability still a customer')
    plt.title("Kaplan-Meier");
    ax.set_ylim([0.0, 1.0])
plt.tight_layout()



#WEIBULL distribution
'''Relies on Weibull statistical distribution, which is 
a continuous probability distribution used to analyse life data, 
model failure times and access product reliability'''

ax = plt.subplot(111)
typ = (df2["CustomerType"] == "A")
wbf.fit(T2[typ], event_observed=E2[typ], label="Cohort A")
wbf.plot_survival_function(ax=ax)
wbf.fit(T2[~typ], event_observed=E2[~typ], label="vs. Rest")
wbf.plot_survival_function(ax=ax)
plt.title("Weibull: Survival function by Customer Type");
plt.xlabel('Time since contract start')
plt.ylabel('Probability still a customer')
ax.set_ylim([0.0, 1.0])
wbf.survival_function_.to_excel("wbf_types.xlsx")

df2 = df.copy()
df2 = df2[df2['ContractAge']>0]
from lifelines import WeibullFitter
T2 = df2['ContractAge']
E2 = df2['Churned']
wbf = WeibullFitter().fit(T2, E2, label="all_types")
wbf.survival_function_.plot()
plt.title('Weibull: Survival function of ALL subscribers');
plt.axhline(0.5, color='red', alpha=0.75, linestyle=':')
plt.xlabel('Time since contract start')
plt.ylabel('Probability still a customer')
ax.set_ylim([0.0, 1.0])
print(wbf.median_survival_time_)
wbf.survival_function_.to_excel("wbf.xlsx")

#COX regression
"""Survival Regression involves utilizing not only the duration and the censorship variables 
but using additional data (Gender, Age, Salary, etc) as covariates. 
We ‘regress’ these covariates against the duration variable.
E.g. Cox Proportional Hazards Regression Analysis Model was introduced by Cox and it takes into account the effect of 
several variables at a time[2] and examines the relationship of the survival distribution to these variables"""

from lifelines import CoxPHFitter
df_stripped = df.drop(['ContractStartDate','ChurnDate','LifespanMonths','Tenure'],inplace=False, axis=1)
cat_col = df.select_dtypes(include=['object']).copy().columns
df_dummy = create_dummy_df(df_stripped,cat_col)
cph = CoxPHFitter(penalizer=0.1)   ## Instantiate the class to create a cph # object
plt.title("Cox Regression");
cph.fit(df_dummy, 'ContractAge', event_col='Churned')   ## Fit the data to train the model
cph.print_summary()    ## Have a look at feature significance
cph.plot();

#The summary statistics above indicates the significance of the covariates in predicting the churn risk
type_b = df_dummy.query('CustomerType_B ==1')
ax = cph.predict_survival_function(type_b).plot(legend=None)
plt.xlabel('Time since contract start')
plt.ylabel('Probability still a customer')
plt.axhline(0.5, color='red', alpha=0.75, linestyle=':') # median line in red
plt.title('Survival curves at customer level: B');
ax.set_ylim([0.0, 1.0])

type_c = df_dummy.query('CustomerType_C ==1')
ax = cph.predict_survival_function(type_c).plot(legend=None)
plt.xlabel('Time since contract start')
plt.ylabel('Probability still a customer')
plt.axhline(0.5, color='red', alpha=0.75, linestyle=':') # median line in red
plt.title('Survival curves at customer level: C');
ax.set_ylim([0.0, 1.0])
