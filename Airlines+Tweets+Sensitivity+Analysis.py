
# coding: utf-8

# In[51]:


#Importing libraries for EDA
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
get_ipython().magic('matplotlib inline')
matplotlib.rcParams.update({'font.size': 12})


# In[52]:


df = pd.read_csv('D:/Personal Projects/EDA Airlines Sentiment Analysis/Tweets.csv')


# In[53]:


df.head()


# In[54]:


#Checking the data structure
df.count()


# In[55]:


#Max count is 14640. The field that has less than 14640 entry has null values in it
#Checking null values per field
df.isnull().sum()


# In[56]:


#Finding different types of sentiments
df.airline_sentiment.unique()


# In[57]:


#Entry per sentiment
df.groupby('airline_sentiment').size()


# In[58]:


#Plotting bar chart of all sentiments
df.groupby('airline_sentiment').size().plot.bar()
plt.ylabel('Sentiment')
plt.title('Overall Airlines Sentiment Picture')


# In[73]:


#Checking datatype of field airline_sentiment
df['airline_sentiment'].dtype


# In[85]:


#Creating a new column for every unique sentiments
df['negative'] = np.where(df['airline_sentiment']=='negative', 1, 0)
df['positive'] = np.where(df['airline_sentiment']=='positive', 1, 0)
df['neutral'] = np.where(df['airline_sentiment']=='neutral', 1, 0)


# In[87]:


#Checking number of unique airlines
df['airline'].unique()


# In[92]:


#Total number of review per airlines
df.groupby('airline').airline.count().plot.bar()


# In[98]:


#Number of negative reviews per airline
df.groupby('airline').negative.sum().plot.bar()
plt.suptitle('Negative reviews per airlines')


# In[100]:


#Neutral review per airline
df.groupby('airline').neutral.sum().plot.bar()
plt.suptitle('Neutral review per airline')


# In[102]:


#Positive review per airline
df.groupby('airline').positive.sum().plot.bar()
plt.suptitle('Positive review per airline')

