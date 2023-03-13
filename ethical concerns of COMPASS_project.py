#!/usr/bin/env python
# coding: utf-8

# ## Title:  Discussion on the ethical concerns of COMPASS

# # Content of the document
#  1. Problem Statement 
#  2. Data Collection
#  3. Data Preparation 
#  4. Data Preprocessing   
#  5. Model Assessement (Fairness) & optimization
#  7. Conclusion
#  8. References 

# # 1. Problem Statement
# 
# We are investigating COMPAS, a popular commercial algorithm for predicting reoffending likelihoods (recidivism) for criminal defendants. There has been evidence that the algorithm is biased towards white defendants and against black inmates. Next, use a fairness approach to improve the model.
# 
# In order to accomplish this, I need first to explore the data and prepare it, then assess the fairness, then use one of the approaches to optimize it

# Here are the steps I will take in this project: 
#     
#      1. Data Collection
#      2. Data Exploration: This will be done to identify the most important features and combine them in new ways.
#      3. Data Preprocessing: Lay out a pipeline of tasks for transforming data for use in my machine learning model.
#      4. Model Assessment: Determine the type of descrimination.
#      6. How to improve the fairness 
#      7. Conclusion & recommendations     

# # 2. Data Collection
# 
# In this step I do two things: 
#     
#   - Identify data sources
#   - Split the data into training and test sets
# 
# 
# Before starting, as a first step, I will call some libraries I need in order to build my model.
# 

# In[1]:


# Libraries 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import set_matplotlib_formats
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import warnings
from sklearn.model_selection import StratifiedShuffleSplit
warnings.filterwarnings("ignore")
from sklearn.datasets import make_regression, make_classification, make_blobs
import sklearn.model_selection
import sklearn.linear_model


# #### Source of the data: (kaggle.com, n.d.)

# In[3]:


# Load the data from Kaggle Repository
initial_data = pd.read_csv('cox-violent-parsed.csv')

# Examine date structure and return the top 5 rows of the data frame.
initial_data.head(5)


# In the table above, the date columns are displayed in object format

# In[4]:


# Convert date columns from object format to time format

# attributes is the list of columns to be converted
attributes= ["in_custody","out_custody","v_screening_date","compas_screening_date", "dob","c_jail_in","c_jail_out","c_offense_date",
             "screening_date","vr_offense_date","r_jail_out","r_jail_in","r_offense_date","c_arrest_date"]

initial_data = pd.read_csv('cox-violent-parsed.csv',parse_dates=attributes)

# Create a copy of the original data
my_data = initial_data.copy()

# Examine my data structure and return the top 5 rows of the data frame.
my_data.head(5)


# In[5]:


#Check the type data of my attributes
my_data.info()


# In[6]:


#Check if there are null values in my dataset
my_data_not_nut = my_data.isnull().sum()
#display non null data
my_data_not_nut[my_data_not_nut>0]


# 48% (25 from 51) of the attributes has null values!

# The dataset I have has 52 attributes, to make my analysis more efficient I will drop the unecessary ones to my study

# In[7]:


#List of the attributes I am keeping
needed_attributes = ["id","name","dob","compas_screening_date","c_offense_date","sex","age","age_cat","race","c_charge_degree","c_charge_desc", 
"days_b_screening_arrest", "decile_score", "is_recid","r_offense_date", "c_case_number","v_decile_score", 
"is_violent_recid","vr_offense_date","score_text"] 

#Copy my data in a new variable, to keep the original one untouched
analysis = my_data.loc[:,needed_attributes].copy()
analysis


# In[8]:


# Check if my dataset has any duplicates for the same name

analysis.duplicated().sum()
analysis[analysis.duplicated(["name","age","sex","race","dob"],keep=False)]


# The data has many multiple duplicates. 

# In[9]:


# remove duplication of the same case, every case should be represented one time
analysis_ND=analysis.drop_duplicates(subset='c_case_number',keep='last')
analysis_ND


# after applying drop duplicate, 1123 Rows deleted from the dataset, now I have only 10310 case remaining 

# My next step would be prepare my data for analysis

# # 2. Preparing dataset for analysis

# Before starting to prepare my data, I remove both the attributes name and dob 

# In[10]:


# Drop the attributes name and dob because they will not impact my study 
analysis_ND = analysis_ND.drop("dob", axis=1)
analysis_ND = analysis_ND.drop("name", axis=1)
analysis_ND


# 1. Keep only rows that has a case number and has an id 

# In[11]:


# keep only rows with a case number 
analysis_ND = analysis_ND[analysis_ND["c_case_number"] != "NaN"]


# In[12]:


# remove all rows with any NaN and NaT values
#analysis_ND =  analysis_ND.dropna()
analysis_ND = analysis_ND.dropna( how='any', subset=['id'])
analysis_ND 


# 2. I want explore how long people take to be charge for a crime and see if I can filter my selection for cases

# In[13]:


#Check how long arrested people take to be charged for a crime
analysis_ND["days_b_screening_arrest"].describe()


# More than 75% have less than -1 and with -597 as minimum
# to keep only quality data, we will limit my analysis on crimes charger within 30 days of arrest

# In[14]:


# Keep only crimes charged within 30 days of arrest.
analysis_ND = analysis_ND.loc[(analysis_ND["days_b_screening_arrest"] > -30) & (analysis_ND["days_b_screening_arrest"] <30)]


# 3. Since my purpose is evalute Compass, I will remove all the cases that has the crime after the compass screening 

# In[15]:


# Remove cases where the date of crime is after copmas screening
analysis_ND = analysis_ND[analysis_ND["c_offense_date"] < analysis_ND["compas_screening_date"]]


# 4. Now, I will create 3 groups: Age, Race and Score using CategoricalDtype

# In[16]:


# Group 1: Age
grp_age = pd.CategoricalDtype(categories=["Less than 25","25 - 45","Greater than 45"],ordered=True)
analysis_ND["age_cat"] = analysis_ND["age_cat"].astype(grp_age)

# Group 2: Race
grp_race = pd.CategoricalDtype(categories=['African-American','Caucasian','Hispanic',"Other",'Asian',
'Native American'],ordered=True)
analysis_ND["race"] = analysis_ND["race"].astype(grp_race)

# Group 3: Score
grp_score = pd.CategoricalDtype(categories=["Low","Medium","High"],ordered=True)
analysis_ND["score_text"] = analysis_ND["score_text"].astype(grp_score)


# In[17]:


#convert the attributes between "Sex" and "c_charge_degree" to categories with the function astype

for att in ["sex","c_charge_degree"]:
    analysis_ND[att] = analysis_ND[att].astype("category")


# 5. Remove all the rows that has empty value for the attribute score_text

# In[18]:


#remove rows with score text equals NaN 
analysis_ND = analysis_ND[analysis_ND["score_text"] != "NaN"]


# In[19]:


#I will not consider traffic tickets & munipal violations as repeated offense (charge degree equals O)
#analysis_ND = analysis_ND[analysis_ND["c_charge_degree"] != "O"]


# 6. I want to know since when the Compass screening happened for both ordinary and violent offenses (I call both variables 2y_r and 2y_v), to do  that I first substract
# the columns of time, then I assign 3 different values for every case:  
#     - 0: the offender didnot commit any crime since 1st arrest 
#     - 1: the offender committed a new crime in less than 2 years
#     - 2: the offender committed a new crime after 2 years    

# In[20]:


# I will call my function nbr_offenses
def nbr_offenses(att,recid):
   
    # Subtract the columns of time
    analysis_ND["days"] = analysis_ND[att] - analysis_ND["compas_screening_date"]
    
    # Convert the output to an integer by using the .days parameter  
    analysis_ND["days"] = analysis_ND["days"].apply(lambda x:x.days)
    
    # Assign the values 0, 1 and 2 
    analysis_ND["offence"] = np.where(analysis_ND[recid]==0,0,
                np.where((analysis_ND[recid]==1) & (analysis_ND["days"] < 730),1,2))
    
    return analysis_ND["offence"]


# In[21]:


# calcualte both 2y_r and 2y_v
analysis_ND["2y_r"] = nbr_offenses("r_offense_date","is_recid")
analysis_ND["2y_v"] = nbr_offenses("vr_offense_date","is_violent_recid")


# 7. I want to foucs only on offenders who didnot or did repeat crime in less then 2 years, so I will remove Offenders who has 2y_r and 2y_v equals 2

# In[22]:


# remove offenders who committed new ordinary crimes after 2 years
analysis_nd_r = analysis_ND[analysis_ND["2y_r"] !=2].copy()

# remove offenders who committed new violent crimes after 2 years
analysis_nd_v = analysis_ND[analysis_ND["2y_v"] != 2].copy()


# In[23]:


# reset the index to  make it easier for me to work with the dataset 
analysis_nd_r.reset_index(drop=True,inplace=True)
analysis_nd_v.reset_index(drop=True,inplace=True)


# 8. In order to make data extraction easier, I add another 2 new attributes:
#     - <b>Attribute 1 :</b> devide cases in 2 categories based on the score using binary values,
#     so if the score >5, the case category will be 1, if the score is lower than 5 the case category will be 0
#     - <b>Attribute 2 :</b> This attribute will help me know if the prediction of Compass was correct or wrong,
#         so if it is "True", the prediction of Compass was correct, if "False" the prediction was wrong

# In[24]:


#Attribute 1: grouping cases in 2 categories (high score (>=5 and low score <5))
analysis_nd_r["binary_score"] = np.where(analysis_nd_r["decile_score"] >=5,1,0)
analysis_nd_v["binary_v_score"] = np.where(analysis_nd_v["v_decile_score"] >=5,1,0)


# In[25]:


# Attribute 2: if the prediction is correct return True,else return false

# For ordinary offense
analysis_nd_r["prediction_recid"] = analysis_nd_r['is_recid'] == analysis_nd_r["binary_score"]

# For violent offense
analysis_nd_v["prediction_vrecid"] = analysis_nd_v['is_violent_recid'] == analysis_nd_v["binary_v_score"]


# In[26]:


# Index reset 
analysis_nd_r.reset_index(drop=True,inplace=True)
analysis_nd_v.reset_index(drop=True,inplace=True)


# In[27]:


#Checking the new values
analysis_nd_r.head(3)


# In[28]:


analysis_nd_v.head(3)


# #### Now my data is ready for the exporatory analysis

# # 3. Data Exploration
# In this step, I will be exploring my data to understand more about: the relationship between the crime, race, gender, age and the frequent commited crimes

# In[29]:


# check data
analysis_nd_r.describe().T


# As shown in  the table above, from the ordinary 3527 cases: 
#    - 75% of the cases assessed by compass got score 6
#    - 75% of the offenders are above 42 years old where 32% are recidivists and 7 % are violent recidivist

# In[30]:


analysis_nd_r.describe(include=["object","category"]).T


# From the table above, we notice:
#  - One of the most common cases is Battery with charge degree F3 with the main race African Americans      
#  - TMost reported cases are from males aging between 25 and 45 

# <b> After cleaning data, and preparing it, now I will apply SVM to classify it

# ## 3.1 Explore Race

#  {change me}
#   - th race, when looking at the statistics, we must always keep in mind that there is very little data on Asians and Native Americans (a good idea would be to group them together with Other in a single category to make it more representative).
# 
#  - In general, we realize that almost 50% of the cases belong to African Americans and they are also the ones with the highest percentage of recidivism (almost 40% of the cases are African American).
# 
#  - This is important because the dataset is not balanced and can be very biased due to the predominance of this ethnic group.

# In[31]:


race_g = analysis_nd_r["race"].value_counts(normalize=True,ascending=False).reset_index()

plt.figure(figsize=(8,4))
sns.barplot(x=race_g["index"],y=race_g["race"],color='#42b7bd')
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("% of cases")
plt.title("% of cases by race")
plt.show(block=False)


#  - 50% of the cases belong to African-Americans, 35% to Caucasian, and the rest is below 1%   

# In[32]:


analysis_nd_r.groupby("race",as_index=False)["2y_r"].mean().style.background_gradient(cmap='Reds',axis=0)


#  - For Recidivism rate, 39 % of African Americans are recidivist and for Caucasian 28% 

# In[33]:


##Obtain percentages by rows with normalize
pd.crosstab(analysis_nd_r["race"],analysis_nd_r["age_cat"],normalize=0).style.background_gradient(cmap='Oranges',axis=1)


# - From the table above, we notice that the predominant age for the different races is between 25 and 45

# ## 3.2 Explore link between sex and age

# In[34]:


plt.figure(figsize=(8,5))
sns.countplot(data=analysis_nd_r,x="sex",hue="age_cat",palette="Greens_r")
plt.title("Cases by age & sex")
plt.xlabel("")
plt.show(block=False)


# Females are secondary compared to males

# In[35]:


decile_mean_age = analysis_nd_r.groupby("age_cat")["decile_score"].mean().reset_index()
sns.barplot(data=decile_mean_age,x="age_cat",y="decile_score",palette="Greys_r")
plt.ylabel("Decile score mean")
plt.xlabel("")
plt.title("Average decile_score by different age ranges")
plt.show(block=False)


# - There is a correlation between age and decile score, the younger the person, the more penalized is.

# ## 3.3 Type of crime per sex (gender)

# I focus, because of their great variety, only on the 10 most typical.
# Many could be grouped by type of crime (drugs, vehicle, robbery and assault...) to facilitate the study of the data.

# In[36]:


#Top 5 crimes for women
female_crime = analysis_nd_r[analysis_nd_r["sex"]=="Female"]["c_charge_desc"].value_counts(normalize=True,ascending=False)[:5].reset_index()
sns.barplot(data=female_crime,x="c_charge_desc",y="index",palette="Reds_r")
plt.title("Top 5 crimes for females")
plt.ylabel("")
plt.xlabel("Cases percentage")
plt.show(block=False)


# In[37]:


#Top 5 crimes for men
male_crime= analysis_nd_r[analysis_nd_r["sex"]=="Male"]["c_charge_desc"].value_counts(normalize=True,ascending=False)[:5].reset_index()

sns.barplot(data=male_crime,x="c_charge_desc",y="index",palette="Greens_r")
plt.title("Top 5 crimes for females")
plt.ylabel("")
plt.xlabel("percentage of cases")
# so only the graphic appears without any text referring to the object type.
plt.show(block=False)


#  Men and women commit the same top 3 offenses

# ## 3.4 Top offenses by both race and sex

# In[38]:


# Here I only consider the top 5 cases by counting their valuses then filtering with the function isin
top_5_cases = analysis_nd_r["c_charge_desc"].value_counts()[:5].index.tolist()
top_cases = analysis_nd_r[analysis_nd_r["c_charge_desc"].isin(top_5_cases)]


# Lets explore in more depth top crimes by race

# In[39]:


pd.crosstab(index=top_cases["c_charge_desc"],columns=top_cases["race"],normalize=1)    .style.background_gradient(cmap='Reds',axis=0)


# - We already saw from the previous sections, Battery is top committed crime, but what we know more now is that its is taking the highest % for the different races. 
# - For the top cases, we still have the sames result. 
# - The exception we see that both Asian and Native American don't have any crimes on the category Driving while Licesne revoked and also no Grand Theft.  

# In[40]:


# Recidivism % between race and the type of the case 
pd.pivot_table(data=top_cases,values="2y_r",index="c_charge_desc",columns="race",fill_value=0)    .style.background_gradient(cmap='Oranges',axis=1)


# In[41]:


# % of recidivism by crime type
percentage_recid_cases= top_cases.groupby("c_charge_desc")["2y_r"].mean().reset_index().sort_values(by="2y_r",ascending=False)

plt.figure(figsize=(8,4))
sns.barplot(data=percentage_recid_cases,y="c_charge_desc",x="2y_r", color="#7FFFD4")
plt.title(" Recidivism by crime type\n",fontsize=15)
plt.xlabel("")
plt.ylabel("")
plt.axvline(x=0.34,color="#4169E1")
plt.show(block=False)


# ## 4. Check the fairness of the model & how to improve it
#  - I will compare both ordinary & violent offences and see wich one has better accuracy using logisitc linear 
#  - I will also see the result by both race and sex

# In[42]:


# first I will calcualte TPR, FPR, FNR, TNR for normal offences
from sklearn.metrics import classification_report
print(classification_report(analysis_nd_r['2y_r'],analysis_nd_r["binary_score"]))


# In[43]:


# calcualte TPR, FPR, FNR, TNR for normal offences
cross_tab = pd.crosstab(analysis_nd_r["binary_score"],analysis_nd_r['2y_r'],normalize="columns").style.background_gradient(cmap='Reds',axis=1)
cross_tab


# - This model has an accuray of 66%
# - Of the actual positive cases it predicted 61% where it is correct in 48% of the cases => False Positive Rate : 1-0.68 = 0.32 
# - Of the actual negative cases, the model predicted 68% of the cases where it was correct for the 78% cases 
#   => False Negative Rate 1-0.61=0.39

# Now I check the prediction for violent crimes

# In[44]:


# analysis_nd_v only cases with violent crimes
pd.crosstab(analysis_nd_v["binary_v_score"],analysis_nd_v['2y_v'],normalize="columns").style.background_gradient(cmap='Greens',axis=1)


# In[45]:


print(classification_report(analysis_nd_v['2y_v'],analysis_nd_v["binary_v_score"]))


# The prediction here is worse than the previous one even though the accuracy shows a higher score,
# as we see, the prediction for positive cases is higher compared to the previous one (18% more) 

# In[46]:


analysis_nd_v['2y_v'].value_counts(normalize=True)


# In[47]:


analysis_nd_r["2y_r"].value_counts(normalize=True)


# - Despite the accuracy being higher (0.70 > 0.66) 
# - In case of violent crimes, the model is less accurate for positive cases (diffirence 18% higher) 
# - False Positive Rate : 0.46
# - False Negative Rate:  0.28
# 
# Conclusion, it shows more accuracy with less accurate predicition for positive cases 

# ## 4.1 Check Descrimination of COMPASS for both the race and sex

# In[48]:


# Comparaison between Revidivism &  Compas forcast
reality = analysis_nd_r.groupby("race",as_index=False)["2y_r"].mean()
compas_prediction = analysis_nd_r.groupby("race")["binary_score"].mean().reset_index()

sns.barplot(data=compas_prediction,x="race",y="binary_score",color="powderblue",label="Prediction")

ax= sns.barplot(data=reality,x="race",y="2y_r",color="blue",label="Real")

plt.title("Recidivism rate in reality vs. Compas prediction")
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("")
plt.legend()
plt.show(block=False)


# Comparing the reality with the system COMPASS, shows hi less accuray for false positive prediction where the highest % goes for Afriacan american followed by Asians and Native Americans 

# In[49]:


analysis_nd_r.groupby("race").agg({"2y_r":"mean",
                                    "binary_score":"mean"}).style.background_gradient(cmap='Greens',axis=1)


# In[50]:


race_acc = analysis_nd_r.groupby("race")["prediction_recid"].mean().reset_index()
ax= sns.barplot(data=race_acc,x="race",y="prediction_recid",color="#FED8B1")
plt.title("  COMPASS accuracy rate by race")
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("")
for num,text in zip(range(6),round(race_acc["prediction_recid"],2)):
    ax.text(num,text-0.05,text)


# In[51]:


# Recidivism by gender
prediction_sex = analysis_nd_r.groupby("sex")["binary_score"].mean()
recidivism_sex = analysis_nd_r.groupby("sex")["2y_r"].mean()
comparacion_sex = analysis_nd_r.groupby("sex")["prediction_recid"].mean()

dt_comp_sex_recidivism = pd.concat([prediction_sex,recidivism_sex,comparacion_sex],axis=1).reset_index()
dt_comp_sex_recidivism.columns = ["sex","decile_score","2y_r","accuracy"]
dt_comp_sex_recidivism.round(2).style.background_gradient(cmap='Greens',axis=1)


# <b>If we look into sex, in real world males are more recidivists than females and we see same in compass. 

# # 4.2 Percentage of false positives and negative positives

# In[52]:


# Create 2 attributes one for the name and one for the result
def wrong_prediction(att):
    # Build 2 lists one for the FPR and one for FNR
    list_att = []
    FPR = []
    FNR = []
    for x in analysis_nd_r[att].unique().tolist():    
        #Filter by race or gender
        data = analysis_nd_r[analysis_nd_r[att]==x] 
        # create sorting report (with output_dict we return a dictionary)   
        classif_race = classification_report(data['2y_r'],data["binary_score"],output_dict=True)
        list_att.append(x)
        # False Positive Rate is 1-exahusivity(recall)
        false_positive = 1 - classif_race.get("0")["recall"]
        FPR.append(false_positive)
        # False Negative Rate 1-TPR
        false_negative = 1 - classif_race.get("1")["recall"]
        FNR.append(false_negative)
    # creamos dataframe
    df_fpr = pd.DataFrame({x:list_att,"FPR":FPR,"FNR":FNR})
    return df_fpr


# In[53]:


wrong_prediction("race").style.background_gradient(cmap='Oranges',axis=1)


# We can see clearly in the table above, that the African- American has the highest % of False positive rate compared to the rest of races. 
# => <b> This shows clearly the mistreat discrimination 

# In[55]:


wrong_prediction("sex").style.background_gradient(cmap='Blues',axis=1)


# For Gender, Recidivism is same for both gender.  

# In[56]:


d = analysis_nd_r.groupby(["decile_score","race"]).agg({"2y_r":"mean"}).reset_index()
d = d[d["race"].isin(["African-American","Caucasian"])]
im = sns.scatterplot(data=d,x="decile_score",y="2y_r",hue="race")
im.set(ylim=(0,1))
plt.show(block=False)


# # 4.SUMMARY
# 1. Compas evaluates by race and gender; there are some differences in results, but we must not forget that one of the races has the highest prevalence (nearly 50%) and they are also the most likely to recidivate (nearly 40%).
# 2. Adding the two_year_r feature based on compas report was for the purpose to see if adding this feature will help to narrow the prediction coming from COMPASS 
# 3. We saw that is_violent_racid got better compard to is_recid but no improvement on Positive cases prediction it became worse. 
# My suggestion, is to remove the race from the full data and train the model in a way that it will be always independant from the race. In that way the discrimination will be avoided, maybe not fully but at least it would be better.   

# # 9.References

# kaggle.com. (n.d.). Compas: thoroughly investigating the controversial. [online] Available at: https://www.kaggle.com/code/gonzalogarciafuste/compas-thoroughly-investigating-the-controversial/data [Accessed 25 OCt. 2022].
