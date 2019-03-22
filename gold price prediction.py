
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting graphs and chaarts
import seaborn as sns #drawing statistical graphics
get_ipython().run_line_magic('matplotlib', 'inline')
gold_data = pd.read_csv('C:/Intellimind/gld_price_data.csv') #reading the data file
gold_data.head()  
gold_data.isna().sum()
import pandas_profiling


# In[2]:


gold_data


# In[3]:


gold_data.describe()


# In[4]:


gold_data.info()


# In[5]:


pandas_profiling.ProfileReport(gold_data)   # The entire exploartory analysis


# In[6]:


sns.jointplot(x='SLV',y='GLD',kind='hex',data=gold_data)


# In[7]:


sns.jointplot(x='EUR/USD',y='GLD',kind='hex',data=gold_data)


# In[8]:


sns.jointplot(x='USO',y='GLD',kind='hex',data=gold_data)


# In[20]:


sns.jointplot(x='SPX',y='GLD',kind='hex',data=gold_data)


# In[9]:


sns.heatmap(gold_data.corr())


# In[10]:


sns.pairplot(gold_data)


# In[2]:


X = gold_data[['SPX','USO','SLV','EUR/USD']]  #mentioning the predictor variables 
y = gold_data['GLD']   # mentioning response variable
#splitting the train and test dataset
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)  #splitting train and test dataset


# In[3]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)  #training the model with the linear regression function


# In[29]:


lm.coef_  #getting the coefficients after the regression


# In[31]:


print(lm.intercept_) #printing the intercept


# In[32]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df  #coefficients in a matrix form


# In[5]:


predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y') #


# In[34]:


plt.plot(y_test,predictions,linestyle='solid')


# In[6]:


from sklearn import metrics
print('MAE :'," ", metrics.mean_absolute_error(y_test,predictions))
print('MSE :'," ", metrics.mean_squared_error(y_test,predictions))
print('RMAE :'," ", np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# In[15]:


sns.distplot(y_test - predictions,bins=50)


# In[22]:


plt.ion()


# In[4]:


# set standard plot parameters for uniform plotting
plt.rcParams['figure.figsize'] = (10, 6)
# prettier plotting with seaborn
import seaborn as sns; 
sns.set(font_scale=1.5)
sns.set_style("whitegrid")


# In[7]:


type(gold_data['Date'][0])


# In[9]:


# create the plot space upon which to plot the data
fig, ax = plt.subplots(figsize = (100,100))

# add the x-axis and the y-axis to the plot
ax.plot(gold_data['Date'], 
        gold_data['GLD'], 
        color = 'red')

# rotate tick labels
plt.setp(ax.get_xticklabels(), rotation=45)

# set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Gold Price",
       title="Gold Price over months")


# In[16]:


gold_data.shape

