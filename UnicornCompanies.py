#!/usr/bin/env python
# coding: utf-8

# # Unicorn Companies
# ## A data manipulation project with Python.
# 
# The dataset gather severeal billionare companies and show when they've became that, how much invested and by who.  
# The project is developed to answer to the following questions:  
# 
# + Which unicorn companies have had the biggest return on investment?
# 
# + How long does it usually take for a company to become a unicorn? Has it always been this way?
# 
# + Which countries have the most unicorns? Are there any cities that appear to be industry hubs?
# 
# + Which investors have funded the most unicorns? 
# 
# <span style="color:blue">*To scroll slides press right on keyboard, to view sub-slide press down. Esc for global view*</span>.

# Dataset is from [Kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/unicorn-companies?datasetId=2192732&sortBy=dateRun&tab=profile)
# ![Immagine.jpg](attachment:Immagine.jpg)

# In[1]:


# Libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import regex as re

# Read dataset
df = pd.read_csv("Unicorn_Companies.csv", delimiter=",")


# In[867]:


df.shape


# In[868]:


df.head(11)


# In[869]:


df.describe() # only two quantitative


# In[870]:


df.dtypes # Type of variables


# In[871]:


df.columns


# ### Null Values
# Variables like City and Select Investors has null values. The command to show them is df.isna()  
# Below is shown a clear summary:

# In[6]:


pd.DataFrame(df.isna().value_counts()) # check null value in each column


# In[873]:


# na values
df.isnull().sum() # 16 city null and 1 select investor null


# In[875]:


# show where city is null
citynull = np.where(df['City'].isnull())
df.iloc[citynull]


# In[876]:


citynull = df.iloc[citynull] # new df with only city:NAN
citynull[["Country"]]


# In[877]:


countrywithnll = citynull.Country.unique()


# After some manipulation, it's shown all the cities in each country that has at least one NaN value.   
# Singapore has only NaN city like Bahamas. Hong Kong is mixed with NaN and cities. 

# In[878]:


# show for every country with a null value all the cities 
for i in countrywithnll:
    asd=df[df.Country.isin([i])]
    print(asd[['City', 'Country']])


# ### Typing error
# A specific view about Industry variable shows an error of typing for Artificial Intelligence (first two rows)

# In[879]:


# industry
sorted(list(df['Industry'].unique()), reverse = False) # there is AI repeated


# In[7]:


df['Industry'] = df['Industry'].replace(['Artificial intelligence'],'Artificial Intelligence') # renaming


# Now I can count rows for each Industry type. 

# In[8]:


# Industry analysis
df['Industry'].value_counts() # 15
#df['Industry'].value_counts()['Fintech']


# In[10]:


ind = df['Industry'].value_counts()
ind = pd.DataFrame(ind)
ind['%']= ind / ind.sum() * 100
print(ind) # to show percentage


# In[49]:


qwert = df['Industry'].value_counts().index
explode = [0, 0, 0, 0, 0,
          0,0,0,0,0,
          0,0.5,0.5,0.5,0.5]

fig, ax = plt.subplots(figsize=(10, 5))
ax.pie(df['Industry'].value_counts(), 
      labels = qwert,
      autopct='%.0f%%',
      explode = explode)
plt.show() # pie chart


# Top billionaire companies' Cateogries are involved with internet. Fintech, Internet services and E-commerce are the most prolific areas. 

# # Where are located most of unicorns?  
#   
# Data Analysis proceed with Continent.

# In[884]:


# continent analysis
continent = pd.DataFrame(df['Continent'].value_counts()) # converto in df e conto i valori 
continent['%'] = continent / continent.sum() * 100
print(continent) # Norh America is the greatest with 54%


# In[885]:


f, ax = plt.subplots(figsize=(10, 5)) # for plot dimensions !!
sns.despine(f) # remove cartesian's lines o top and right

sns.histplot(df,
            x = 'Continent',
            multiple="stack",
            palette="light:m_r",
            edgecolor=".3",
            linewidth=.5)


# # Analysis between valuation and funding
# 
# Is there a correlation between these two variables? 

# In[2]:


# renaming column
# df.rename(columns={"Valuation":"Valuation"}, inplace= True) 
# inplace is fundamental because this command create, by default, a new df with renamend column
# with inplace I told him to modifiy existing one

# Now remove B and $ from values in Valuation and conversion to int
df['Valuation'] = df['Valuation'].str.replace(
    "B","", regex= True).str.replace(
    "$", "", regex = True) # command str.replace is for replacing values in columns
df['Valuation'] = df['Valuation'].astype(int)

# otherway
#df['Valuation'] = np.where(df.Valuation.str.contains("B"), "", df['Valuation']) # good to know
# 'where' wants three argument: condition, what to do if it's true, what else

# pd.to_numeric(df['Valuation'], inplace = True) # conversion in numeric


# In[887]:


df.Valuation.unique()


# In[3]:


# using "where" I selected values in Million and added "0," at the start!!
df['Funding'] = np.where(df['Funding'].str.endswith("M"),
                         df["Funding"].str.replace("$","0,", regex = True), 
                         df["Funding"].str.replace("$","", regex = True))
df['Funding'] = df['Funding'].str.replace("B", "", regex = True).str.replace("M","", regex = True)# removed "B" and "M"
df['Funding'] = df['Funding'].replace("Unknown", np.nan) # added NaN instead of unknown
df['Funding'] = df['Funding'].str.replace(",", ".", regex = True).astype(float) # converting string to float (ex. 1,468)
df['Funding'] = df['Funding'].round(2)


# In[4]:


df['Funding'].unique()


# In[890]:


df.Funding.isnull().value_counts()


# In[891]:


df.Funding.describe()


# After renaming variables and modifications about correct unit of measure, it's shown unicorns with funding values equal to NaN

# In[892]:


fundingNull = df[df.Funding.isnull()] # created df with all Null value of Funding
fundingNull # lets ignore it


# In[893]:


f, ax = plt.subplots(figsize=(10, 5)) # for plot dimensions !!
sns.despine(f) # remove cartesian's lines o top and right

sns.scatterplot(data = df,
               x = "Valuation",
               y = "Funding",
               hue="Continent")
plt.xlim(0,25) # from 0 to 25


# In[894]:


f, ax = plt.subplots(figsize=(10, 5)) # for plot dimensions !!
sns.despine(f) # remove cartesian's lines o top and right

sns.scatterplot(data = df,
               x = "Valuation",
               y = "Funding",
               hue="Continent")
plt.xlim(left=25)


# It doesn't seem being a relationship between Funding and Valuation. We expected a positive relation but it's not so clear.

# In[895]:


df.corr()


# In[896]:


df.Valuation.corr(df.Funding)


# Correlation is slightly more than 50% so nothing can be said about.

# In[6]:


f, ax = plt.subplots(figsize=(10, 5))
sns.despine(f)
sns.regplot(data=df, x="Valuation", y = "Funding",)


# # More prolific Cities
# 
# Examination of cities with more billionaire companies 

# In[897]:


df.City.describe() # useful description by pandas


# In[898]:


# SKIP
asd = df.City.value_counts() 
asd = pd.DataFrame(asd)
asd["Perc"] = asd / len(asd) * 100
asd["Perc"] = sorted(asd.Perc, reverse = True) # ordered descending 
for i in range(len(asd["Perc"])-1): # removing with if inside a for loop cities under 3 %, it's only a try
    if asd["Perc"].iloc[i+1] < 3:
        asd["Perc"].iloc[i+1] = 0
    else:
        i = i + 1 
        
asd.head(30)
#---------


# In[899]:


qwe = pd.DataFrame(df.City.value_counts().head(20))
qwe["Perc"] = qwe / len(qwe) * 100
qwe["Perc"] = sorted(qwe.Perc, reverse = True)
label = qwe.axes[0] # to extract row indexes

plt.subplots(figsize=(15, 10))
plt.pie(qwe.Perc, 
      labels = label,
      shadow = True)
plt.legend(title = "Top innovation Cities", 
           loc = 9)
plt.show() # pie chart


# As showed by Country: North America has the two biggest innovation cities: NY ans SanF, followed by China with Beijing and Shangai.

# # Which unicorn companies have the biggest return on investment?
# 
# To determine ROI is required a column which enhance this expression: 
# 
# ```
# ROI = (Valuation - Funding) / Funding * 100
# ```

# In[900]:


roi = pd.DataFrame(df.loc[:,["Company", "Valuation", "Funding", "Date Joined", "Year Founded"]])
# the biggest ROI ignoring time
# I have to calculate Net Profit by decreasing to Valuation the Funding
# then divide by Funding * 100
roi['NetProfit'] = roi.Valuation - roi.Funding
roi['ROI'] = roi.NetProfit / roi.Funding
roi['ROI'] = (roi.ROI * 100).round(2)
#roi["roi"] = (((roi.Valuation - roi.Funding) / roi.Funding) * 100).round(2) # why doesn't workkkkk?
#roi.sort_values('ROI', ascending = False)
roi.sort_values('ROI', ascending=False).head(11)


# In[901]:


na = np.where(roi.ROI.isnull())
roi.loc[na] # ok, every NaN is due from Funding == NaN


# In[902]:


sns.relplot(x="Funding", y="ROI", size="Valuation",sizes=(40, 400), alpha=.5, palette="muted", height=6, data=roi)


# Most of ROIs is near 0 with peaks over 4000%

# # How long does it usually take for a company to become a unicorn? 

# In[874]:


# change in date type
df['Date Joined'] = pd.to_datetime(df['Date Joined']) 


# In[903]:


df['Year Founded'] = pd.to_datetime(df['Year Founded'], format= "%Y") # converted in date time with 01/01 starting point
diff = df['Date Joined'] - df['Year Founded'] # simple difference
diff_float = diff.astype(str) # convert to string
diff_float = diff_float.str.replace("days","", regex = True).astype(int) # remove days and transform into integer
diff_float = diff_float / 365 # calculate years


# In[904]:


diff_float = diff_float.astype(str).str.slice(0,5).astype(float) # sliced number of characters. required to be string
df["Timing in years"] = diff_float # added to df 


# In[905]:


df.sort_values("Timing in years",ascending = True).head() # Yidian Zixun must be ignored


# Table above shows top 5 faster companies to be valued billions.
# The first must be ignored cause there is some issues with date fundation and/or data joined. The other ones are from Norh-America and Asia and were really fast. To explaine this velocity to reach billion valuation we need more information. Probably they are acquired by holdings or were founded as a division of a multinational company so, they already had resources to became so much valuable in so little time.

# In[906]:


df.sort_values("Timing in years",ascending = False).head() # the longest is 98 years


# Above there is another table with the slowest companies. The worst required almost one century but has collected 0,00 Billion $ in funding to became a billionaire.  

# In[907]:


df["Timing in years"].describe() # mean is 7 years


# Mean time is 7 years.

# In[908]:


sns.set_theme(style="white")
sns.relplot(x=df["Timing in years"], y="Funding", hue = "Industry", size="Valuation",sizes=(40, 400), 
            alpha=.5, palette="muted", height=6, data=df)
#plt.xlim(0,30) # X limit
#plt.ylim(0,6) # Y limit


# Great amount of billionaire companies has collected less than 1 billion to became billionaire. 
# Industry is really mixed, there isn't a visible trend.

# In[909]:


# df[df.Company == "Tesla"] # how to search a specific value in a column


# # Which investors have funded the most unicorns?

# In[7]:


df = df.rename(columns={'Select Investors': 'Investors'}, inplace=False)


# In[12]:


pd.DataFrame(df.Investors.unique()).rename(columns={0:'Investor'}, inplace=False)


# Output above is shown only to see that column (renamed) Investors contains from 1 to 4 Venture capitalist per row.

# In[912]:


df.Investors.value_counts(sort = True)


# In[913]:


df[df.Investors == "Sequoia Capital"] # it shows only investor row with only " ", not where it's included between commas


# In[914]:


topInv = pd.DataFrame(df.loc[:,['Valuation', 'Investors']]).sort_values(by= 'Valuation', ascending= False)
# selected Valuation and Investor and ordered by biggest Valuation to lowest
topInv.head(11)


# Column Investors has from 1 to 4 investors separated by comma. I have to split them

# In[915]:


#topInv[["Investor1", "Investor2", "Investor3"]] = topInv.Investors.str.split(pat=",", expand = True)
topInv.Investors.str.split(pat=",", expand = True).iloc[:,[3]].describe() # there are investors with 4 investor
ghj = topInv.Investors.str.split(pat=",", expand = True).iloc[:,[3]]
ghj[3].sort_values(ascending = False).head(15) # yes, there are 4 columns, so..
topInv[["Inv1", "Inv2", "Inv3", "Inv4"]] = topInv.Investors.str.split(pat = ",", expand = True) # created new column
# with name
# NOTE for calling new columns i have to use two square brackets
# topInv = topInv.drop(columns= ['Investors'])


# In[916]:


topInv.Inv2.value_counts()


# In[917]:


topInv.Inv3.value_counts()


# In[918]:


topInv.Inv4.value_counts()


# In[919]:


topInv.iloc[np.where(topInv.Investors.str.contains("Sequoia Capital"))] # splitted
# there are two sequoia capital, with and without China and more..


# Splitted in 4 columns

# In[920]:


553+601+576+8


# In[921]:


# c1 = pd.DataFrame(topInv.Inv1.value_counts())
# c2 = pd.DataFrame(topInv.Inv2.value_counts())
# c3 = pd.DataFrame(topInv.Inv3.value_counts())
# c4 = pd.DataFrame(topInv.Inv4.value_counts())

##### let' find an automated way to do it -------

d = {} # Dictionary
for i in list(range(2,6,1)):
    d['c'+str(i)]=pd.DataFrame(topInv.iloc[:,[i]].value_counts()) # dictionary with c2 to c5
    
antonio = []
for i in list(range(2,6,1)): # list extracted from dict
    pl = d['c'+str(i)]
    antonio.append([pl])
# I'm not able to perform a for loop for concatenating lists


# In[922]:


#a = pd.Index(list(range(0,len(c1),1)))
c1 = pd.concat([antonio[0][0], antonio[1][0], antonio[2][0], antonio[3][0]])
c1.reset_index(level = 0, inplace = True) # add index 
c1 = c1.rename(columns={'Inv1':'investor',0:'count'}) # rename column 
# c1['investor'] = c1.index # useful to extract index!


# After splitting and concatenated them, a specific search shows duplicats not identified by code. Below there is an example.

# In[923]:


c1.iloc[np.where(c1.investor.str.contains("Sequoia Capital"))] # there are duplicates


# In[924]:


c1[c1.investor == "Sequoia Capital"]


# In[925]:


c1[c1.investor == " Sequoia Capital"]


# As shown, in the last two chunks, for "Sequoia Capital" there are different rows because of spaces.

# In[926]:


c2 = c1.groupby(['investor']).sum('count').sort_values('count', ascending= False) # groupby
c2.reset_index(level=0, inplace=True)


# In[927]:


c2.iloc[np.where(c2.investor.str.contains("Sequoia"))] # there are duplicates not got by groupby
# it is for blanket spaces, let's try to solve


# In[929]:


c2.investor = c2.investor.str.replace(" ", "") # removed blank spaces
c2[c2.investor.duplicated()] # now it finds duplicates


# Removed blankets and printed duplicates detected.

# In[930]:


c2 = c2.groupby('investor').sum('count').sort_values('count', ascending = False) # another group by, now should work
c2.reset_index(level= 0, inplace= True) 


# And finally, below, there is a realiable counting of most present investors in unicorn companies dataset

# In[931]:


c2 # excellent


# In[932]:


a = pd.DataFrame(c2.investor.astype(str))


# In[933]:


c2['investor'] = pd.Series(c2['investor'], dtype= "string") # astype doesn't work, it s deprecated


# In[934]:


c2['investor'] = c2.investor.str.replace(r"([a-z])([A-Z])",r"\1 \2")


# In[935]:


c2.head(11)


# The biggest investors shown above are Accel, Andreessen Horowitz, Sequoia Capital with different locations and others.
# 
# The project of data manipulation ends here. It has been useful to understand Python's logic and improve abilities with data anlaysis and visualization. As well as practicing with jupyter notebook and slides creation.

# 
# # *The end*
# <p style="text-align:right;">Patrizio Iezzi</p>

# In[936]:


#for i in range(len(c2.investor.head(11))):
#    topInvestor = pd.DataFrame(df.iloc[np.where(df.Investors.str.contains(str(i)))])
#    print(topInvestor)

