#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import warnings

# models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

#Data Preprocessing
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

#Synthetic Data Generation
from sklearn.datasets import make_classification


# Metrics
from sklearn.metrics import confusion_matrix,accuracy_score,mean_absolute_error,mean_squared_error,r2_score,log_loss,f1_score,jaccard_score

#Table Creation
from tabulate import tabulate

# Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Metrics
from sklearn.metrics import accuracy_score, classification_report, roc_curve


# In[2]:


data = pd.read_csv(r'C:\Users\iran\Desktop\dataset.exel\heart.csv')
data                 


# In[3]:


df = pd.DataFrame(data)
df


# In[4]:


df.head(10).style.set_properties(**{"background-color": "1011","color":"#e9c46a","border": "1.5px solid black"})


# In[5]:


df.tail(10).style.set_properties(**{"background-color": "#121d6b","color":"#e9c46a","border": "1.5px solid black"})


# In[6]:


df.describe()


# In[7]:


#### better describe ##### 


# In[8]:


def describe (df):
    variables =[]
    dtypes =[]
    count =[]
    unique =[]
    missing = []
    min_ =[]
    max_ =[]
    
    for item in df.columns :
        
        variables.append(df[item])
        dtypes.append(df[item].dtype)
        count.append(len(df[item]))
        unique.append(len(df[item].unique()))
        missing.append(df[item].isna().sum())
        
        if df[item].dtypes == 'float64' or df[item].dtypes == 'int64':
            min_.append(df[item].min())
            max_.append(df[item].max())
        else:
            min_.append('Str')
            max_.append('Str')
        

    output = pd.DataFrame({
        'variable': variables, 
        'dtype': dtypes,
        'count': count,
        'unique': unique,
        'missing value': missing,
        'Min': min_,
        'Max': max_
    })    
        
    return output


# In[9]:


desc_df = describe(df)
desc_df


# In[10]:


Styles = [dict(selector = "caption",
               props = [("color", "white"), 
                        ("text-align", "center"),
                       ('font-size', '15pt'),
                       ('background-color', 'black')])]


# In[11]:


desc_df = describe(df)
desc_df = desc_df.style.set_caption('Overview of the dataset').set_table_styles(Styles)
desc_df.set_properties(**{"background-color": "#121d6b","color":"#e9c46a","border": "1.5px solid black"})


# In[12]:


numerical = ['age' ,  'trtbps' , 'chol' , 'thalachh'  ] 


# In[13]:


i = 0
while i<4:
    fig = plt.figure(figsize = [12,3])
    plt.subplot(1,2,1)   
    sns.boxplot(x = 'output',y=numerical[i], data=df , color = 'black') 
    plt.xticks(fontsize=15)
    plt.xlabel('output',fontsize=10)
    i += 1
    
    plt.subplot(1,2,2)
    sns.boxplot(x = 'output',y=numerical[i], data=df , color ='deepskyblue') 
    plt.xticks(fontsize=15)
    plt.xlabel('output',fontsize=10)
    i += 1
    plt.show()


# In[14]:


df.hist(bins=30 , figsize =(20 ,15) , color ="b")
plt.show()


# In[15]:


sns.pairplot(df)


# In[16]:


df.boxplot(rot=90 , figsize = (30,20) )


# In[17]:


plt.figure( figsize = (20, 10))
plt.scatter (df ["chol"] , df["output"])
plt.xlabel ("chol")
plt.ylabel ("output")
plt.grid()


# In[18]:


df[(df["chol"] > 350)]


# In[19]:


plt.figure(figsize =(20,10))
plt.scatter(df["trtbps"] , df["output"])
plt.xlabel("trtbps")
plt.ylabel("output")
plt.grid()


# In[20]:


df[(df["trtbps"] > 170)]


# In[21]:


df.drop(index =[8,101,110,203,223,241,248,260,266,4,28,39,85,96,180,220,246] , inplace=True)


# In[22]:


target_0 = df[df["output"] == 0]
target_0 = pd.DataFrame (target_0)
target_1 = df[df["output"] == 1]
target_1 = pd.DataFrame(target_1)


# In[23]:


target_0.sort_values(by = ['age'] , inplace = True)
target_1.sort_values (by = ['age'] , inplace = True)
print (target_0 . shape)
print(target_1 . shape)


# In[24]:


sns.countplot(x='output', data=df, palette=['black', 'navy'])
plt.title('Distribution of Heart Disease (0: No, 1: Yes)' , fontweight='bold')
plt.show()


# In[25]:


#### pie plot


# In[26]:


correlation_matrix = df.corr()


# In[27]:


plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='BuPu', fmt='.2f')
plt.title('Correlation Matrix' , fontweight='bold')
plt.show()


# In[28]:


num_columns = len(df.columns) -1
num_rows =(num_columns +1) //2

plt.figure (figsize = (15 , 5*num_rows))

for i , column in enumerate (df.columns.drop('output')):
    plt.subplot(num_rows , 3 , i+1)
    sns.kdeplot(data=df[df['output']==0][column] , label = 'output = 0' , fill =True  , color = 'deepskyblue')
    sns.kdeplot(data =df[df['output']==1][column] , label ='output=1' , fill = True  , color ='orchid')
    plt.title(f'KDE plot of {column} grouped by output' , fontweight='bold')
    plt.legend()
plt.tight_layout()
plt.show()
    

    



# In[29]:


heart_disease = df['output'].value_counts()[1]
No_heart_diseas = df['output'].value_counts()[0]
tatal_count = len (df)
heart_disease_percentage = (heart_disease / tatal_count) * 100
No_heart_diseas_percentage = (No_heart_diseas / tatal_count) * 100

pie_values = [heart_disease_percentage , No_heart_diseas_percentage]
colors = ['deepskyblue' , 'orchid']


plt.pie(pie_values , labels=['No Heart Attack', 'Heart Attack'] , autopct='%1.2f%%' ,explode=(0.02, 0.02),colors=colors ) 
plt.title ("No Heart Attack vs Heart Attack" , color = 'black'  , fontsize = 16 , fontweight='bold')





# In[30]:


sns.heatmap (df.describe ().T [['mean']], annot = True, cmap = 'coolwarm', fmt = ".2f")


# In[31]:


#################################################### MODEL LEARNING ########################################################


# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics 


# In[33]:


x = df.drop(['output'] , axis = 1)
y = df["output"].values.reshape(-1,1)


# In[34]:


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2, random_state = 42)


# In[35]:


print ("x train" , x_train.shape)
print ("x test" , x_test.shape)
print ("y train" , y_train.shape)
print ("y test" , y_test.shape)


# In[36]:


LR = LinearRegression ()


# In[37]:


LR.fit (x_train , y_train)


# In[38]:


y_pred = LR.predict(x_test)


# In[39]:


print('mean Absolute Error:' , metrics.mean_absolute_error(y_test , y_pred))
print('mean squared Error :' , metrics.mean_squared_error(y_test , y_pred))
print('Root mean squared Error :' , np.sqrt(metrics.mean_squared_error(y_test , y_pred)))
print('R2score:' , metrics.r2_score(y_test , y_pred))


# In[40]:


def chek (Dimension , Testsize):
    r2 =  0.5764526475068403
    for column in x :
        new_col_name = column + str(Dimension)
        new_col_val = x[column]** Dimension
        x.insert(0 , new_col_name , new_col_val)
        x_train , x_test , y_train , y_test = train_test_split (x ,y ,test_size = Testsize , random_state =42)
        new_LR = LinearRegression()
        new_LR .fit(x_train , y_train)
        y_pred = new_LR.predict(x_test)
        r2_new = metrics.r2_score(y_test , y_pred)
        
        if r2_new < r2 :
            x.drop([new_col_name] , axis = 1 , inplace = True)
        else :
                r2 = r2_new
    print  ('R2 score :', r2)       

chek(3, 0.2)
      


# In[41]:


x


# In[42]:


trtbps_chol = x["trtbps"]*x["chol"]
trtbps_thalachh = x["trtbps"]*x["thalachh"]
chol_thalachh = x["chol"]*x["thalachh"]
age_trtbps=x["age"]*x["trtbps"]
age_chol=x["age"]*x["chol"]
age_thalachh = x["age"]*x["thalachh"]
oldpeak3_thalachh3 = x["oldpeak3"]*x["thalachh3"]
thall3_thalachh3=x["thall3"]*x["thalachh3"]
oldpeak3_thall3 = x["oldpeak3"]*x["thall3"]


# In[43]:


x.insert(0 , "trtbps_chol", trtbps_chol)
x.insert(0 , "trtbps_thalachh", trtbps_thalachh)
x.insert(0 , "chol_thalachh", chol_thalachh)
x.insert(0 , "age_trtbps", age_trtbps)
x.insert(0 , "age_chol", age_chol)
x.insert(0 , "age_thalachh", age_thalachh)
x.insert(0 , "oldpeak3_thalachh3", oldpeak3_thalachh3)
x.insert(0 , "thall3_thalachh3", thall3_thalachh3)
x.insert(0 , "oldpeak3_thall3", oldpeak3_thall3)


# In[44]:


x


# In[46]:


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2, random_state = 42)
LR1 = LinearRegression ()
LR1.fit (x_train , y_train)
y_pred = LR1.predict(x_test)


# In[47]:


print('mean Absolute Error:' , metrics.mean_absolute_error(y_test , y_pred))
print('mean squared Error :' , metrics.mean_squared_error(y_test , y_pred))
print('Root mean squared Error :' , np.sqrt(metrics.mean_squared_error(y_test , y_pred)))
print('R2score:' , metrics.r2_score(y_test , y_pred))


# In[48]:


####### SVM ######


# In[49]:


xS = df.drop(['output'] , axis = 1)
yS = df['output'].values .reshape(-1,1)
yS = yS.ravel()
x_train , x_test , y_train , y_test = train_test_split(xS , yS ,test_size= 0.2 , random_state=42)
svm = SVC (kernel='linear', C=1, random_state=42)
svm . fit(x_train , y_train)
y_preds = svm . predict(x_test)
acc = metrics . accuracy_score(y_test , y_preds)
print(acc)


# In[50]:


##########################           SVM With hyper-parameter            ###########################
svm = SVC()


parameters = {"C":np.arange(1,10,1),'gamma':[0.00001,0.00005, 0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5]}

# instantiating the GridSearchCV object
searcher = GridSearchCV(svm, parameters)

# fitting the object
searcher.fit(x_train, y_train)

print("The best params are :", searcher.best_params_)
print("The best score is   :", searcher.best_score_)

y_pred2 = searcher.predict(x_test)
print("The test accuracy score of SVM after hyper-parameter tuning is ", accuracy_score(y_test, y_pred2))


# In[51]:


########## Random Forest ##########


# In[52]:


xR = df.drop(['output'] , axis = 1)
yR = df['output'].values .reshape(-1,1)
x_train , x_test , y_train , y_test = train_test_split(xR , yR ,test_size= 0.2 , random_state=42)
rfc = RandomForestClassifier(n_estimators=350,criterion='entropy',max_depth=45)
y_train = y_train.ravel()
rfcmodel = rfc.fit(x_train,y_train)
score = rfcmodel.score(x_test,y_test)
print(score)


# In[53]:


##############  Logistic Regression  ###############


# In[54]:


xL = df.drop(['output'] , axis = 1)
yL = df['output'].values .reshape(-1,1)
x_train , x_test , y_train , y_test = train_test_split(xL , yL ,test_size= 0.2 , random_state=42)
log = LogisticRegression()
y_train = y_train.ravel()
logmodel = log.fit(x_train,y_train)
score = logmodel.score(x_test,y_test)
print(score)


# In[55]:


##### ROC Curve ######
y_pred_prob = log.predict_proba(x_test)[:,1]

# instantiating the roc_cruve
fpr,tpr,threshols=roc_curve(y_test,y_pred_prob)

# plotting the curve
plt.plot([0,1],[0,1],"k--",'r+')
plt.plot(fpr,tpr,label='Logistic Regression')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistric Regression ROC Curve")
plt.show()


# In[56]:


###############  Decision TreeClassifier  ################


# In[57]:


xT = df.drop(['output'] , axis = 1)
yT = df['output'].values .reshape(-1,1)
x_train , x_test , y_train , y_test = train_test_split(xT , yT ,test_size= 0.2 , random_state=42)
Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 5,random_state=30)
treemodel =Tree.fit(x_train,y_train)
score = treemodel.score(x_test,y_test)
print(score)


# In[58]:



#################   LogisticRegression after  standardscaler  ##############


# In[59]:


xP = df.drop(['output'] , axis = 1)
yP = df['output'].values .reshape(-1,1)
x_train , x_test , y_train , y_test = train_test_split(xP , yP ,test_size= 0.2 , random_state=42)
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(x_train, y_train)  # apply scaling on training data
Pipeline(steps=[('standardscaler', MinMaxScaler()),('LogisticRegression', LogisticRegression())])
pipe.score(x_test, y_test)


# In[60]:


###############   Gradient Boosting Classifier   ################


# In[61]:


# instantiate the classifier
gbt = GradientBoostingClassifier(n_estimators = 300,max_depth=1,subsample=0.8,max_features=0.2,random_state=42)
y_train = y_train.ravel()
# fitting the model
gbt.fit(x_train,y_train)

# predicting values
y_pred = gbt.predict(x_test)

print("The test accuracy score of Gradient Boosting Classifier is ", accuracy_score(y_test, y_pred))


# In[62]:


###################           Confusion Matrix        ###############


# In[63]:


svm.fit(x_train, y_train)
y_predsvm = svm.predict(x_test)
confusion_matrix(y_test,y_predsvm)


# In[64]:


y_predrfc = rfcmodel.predict(x_test)
confusion_matrix(y_test,y_predrfc)


# In[65]:


y_predtree = treemodel.predict(x_test)
confusion_matrix(y_test,y_predtree)


# In[66]:


y_predlog = logmodel.predict(x_test)
confusion_matrix(y_test,y_predlog)


# In[67]:


y_predgbt = gbt.predict(x_test)
confusion_matrix(y_test,y_predgbt)


# In[68]:


LR_Accuracy_Score = accuracy_score(y_test,y_predlog)
LR_JaccardIndex = jaccard_score(y_test,y_predlog)
LR_F1_Score = f1_score(y_test,y_predlog)
LR_Log_Loss = log_loss(y_test,y_predlog)


# In[69]:


print(f"LR_Accuracy_Score is :{LR_Accuracy_Score},LR_JaccardIndex is :{LR_JaccardIndex},LR_F1_Score is :{LR_F1_Score},LR_Log_Loss is :{LR_Log_Loss}")


# In[70]:


SVM_Accuracy_Score = accuracy_score(y_test,y_preds)
SVM_JaccardIndex = jaccard_score(y_test,y_predsvm)
SVM_F1_Score = f1_score(y_test,y_predsvm)


# In[71]:


print(f"SVM_Accuracy_Score is :{SVM_Accuracy_Score},SVM_JaccardIndex is :{SVM_JaccardIndex},SVM_F1_Score is :{SVM_F1_Score}")


# In[72]:


RFC_Accuracy_Score = accuracy_score(y_test,y_predrfc)
RFC_JaccardIndex = jaccard_score(y_test,y_predrfc)
RFC_F1_Score = f1_score(y_test,y_predrfc)


# In[73]:


print(f"RFC_Accuracy_Score is :{RFC_Accuracy_Score},RFC_JaccardIndex is :{RFC_JaccardIndex},RFC_F1_Score is :{RFC_F1_Score}")


# In[74]:


Tree_Accuracy_Score = accuracy_score(y_test,y_predtree)
Tree_JaccardIndex = jaccard_score(y_test,y_predtree)
Tree_F1_Score = f1_score(y_test,y_predtree)


# In[75]:


print(f"Tree_Accuracy_Score is :{Tree_Accuracy_Score},Tree_JaccardIndex is :{Tree_JaccardIndex},Tree_F1_Score is :{Tree_F1_Score}")


# In[76]:


d = {
     'Tree':[Tree_Accuracy_Score, Tree_JaccardIndex, Tree_F1_Score,'-'],
     'LR':[LR_Accuracy_Score, LR_JaccardIndex, LR_F1_Score,LR_Log_Loss],
     'SVM':[SVM_Accuracy_Score, SVM_JaccardIndex, SVM_F1_Score,'-'],
     'RFC':[RFC_Accuracy_Score, RFC_JaccardIndex, RFC_F1_Score,'-']}
Report = pd.DataFrame(data=d, index = ['Accuracy','Jaccard Index','F1-Score','Log Loss']).T
print(tabulate(Report, headers = 'keys', tablefmt = 'psql'))


# In[ ]:




