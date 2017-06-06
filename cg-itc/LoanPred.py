'''
####################################################################################################
Creator : BOT Master team 
Date    :May 2017 
####################################################################################################
'''
#Load Libraries
from __future__ import division
import pandas as pd
import xgboost as xgb
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

#Functions Starts Here ####################################################################################################
def transform(data,SS):
	List=[]
	data["Married"]=data["Married"].replace(np.NaN,SS["Married"].mode()[0])
	data["Education"]=data["Education"].replace(np.NaN,SS["Education"].mode()[0])
	data["Self_Employed"]=data["Self_Employed"].replace(np.NaN,SS["Self_Employed"].mode()[0])
	data["Property_Area"]=data["Property_Area"].replace(np.NaN,SS["Property_Area"].mode()[0])
	data["Gender"]=data["Gender"].replace(np.NaN,SS["Gender"].mode()[0])
	data["Loan_Amount_Term"]=data["Loan_Amount_Term"].replace(np.NaN,SS["Loan_Amount_Term"].mode()[0])
	data["Credit_History"]=data["Credit_History"].replace(np.NaN,SS["Credit_History"].mode()[0]) 
	data["Dependents"]=data["Dependents"].replace(np.NaN,SS["Dependents"].mode()[0])
	data["Education"]=data["Education"].replace(np.NaN,SS["Education"].mode()[0])
	data["ApplicantIncome"]=data["ApplicantIncome"].replace(np.NaN,SS["ApplicantIncome"].mean())
	data["CoapplicantIncome"]=data["CoapplicantIncome"].replace(np.NaN,SS["CoapplicantIncome"].mean())
	data["LoanAmount"]=data["LoanAmount"].replace(np.NaN,SS["LoanAmount"].mean())
	#Replace value with Imputationrules
	data.loc[(data["Married"]=="Yes"),"Married"]=1
	data.loc[(data["Married"]=="No"),"Married"]=0
	data.loc[(data["Gender"]=="M"),"Gender"]=1
	data.loc[(data["Gender"]=="F"),"Gender"]=0
	data.loc[(data["Self_Employed"]=="Yes"),"Self_Employed"]=1
	data.loc[(data["Self_Employed"]=="No"),"Self_Employed"]=0
	data.loc[(data["Education"]=="Graduate"),"Education"]=1
	data.loc[(data["Education"]=="Not Graduate"),"Education"]=0
	data.loc[(data["Dependents"]=="3+"),"Dependents"]=4	
	data.loc[(data["Property_Area"]=="Rural"),"Property_Area"]=0
	data.loc[(data["Property_Area"]=="Semiurban"),"Property_Area"]=1
	data.loc[(data["Property_Area"]=="Urban"),"Property_Area"]=2
	
	data.apply(pd.to_numeric,errors="ignore")
        data["Netincome"]=data["ApplicantIncome"]+data["CoapplicantIncome"]
	data["LoanPerincome"]=data["LoanAmount"]/data["Netincome"]
	return data

	
#Functions Ends Here ####################################################################################################
#Main Starts here    ####################################################################################################
#Load Data
DTrain = pd.read_csv("train_data.csv")
DTest  = pd.read_csv("test_data.csv")

#Create a super set of data to calculate the correct imputation values
ss=DTrain
ss=ss.append(DTest)

# Data Transformation
DTrain=DTrain[np.isfinite(DTrain["Credit_History"])]

DTrain=transform(DTrain,ss)
DTest=transform(DTest,ss)
DTest=DTest[np.isfinite(DTest["Credit_History"])]

#Rearrange the column positions
feature_columns_to_use = ["Gender","Married","Dependents","Education","Self_Employed","Netincome","Property_Area","Credit_History"]

'''
DTrain.hist()
scatter_matrix(DTrain)
plt.show()
'''

#Sampling 
Train_X=DTrain[feature_columns_to_use ].as_matrix()
Train_Y=DTrain["Loan_Status"].as_matrix()
Test_X =DTest[feature_columns_to_use].as_matrix()
validation_size=0.23
seed=7
scoring = 'accuracy'
X_train,X_Test,Y_train,Y_Test= model_selection.train_test_split(Train_X,Train_Y,test_size=validation_size, random_state=seed)	


#Build models
models=[]
models.append(('LR', LogisticRegression()))
models.append(('XGBoost', xgb.XGBClassifier(max_depth=5, n_estimators=325, learning_rate=0.01)))
models.append(('RandomForest',RandomForestClassifier(n_estimators=100,max_features=3)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('GradientBoosting',GradientBoostingClassifier(n_estimators=100,random_state=seed)))
models.append(('AdaBoost',AdaBoostClassifier(n_estimators=100,random_state=seed)))
results=[]
names=[]
print "Loan Approval Model performance"
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)	


'''
#Uncomment to see the plots
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
'''

# Predict values
ensemble=VotingClassifier(models).fit(X_train, Y_train)
finalpred=ensemble.predict(X_Test)
print("Combined Model Accuracy",accuracy_score(Y_Test, finalpred))
print(confusion_matrix(Y_Test, finalpred))
print(classification_report(Y_Test, finalpred))

#Submission
ensemble=VotingClassifier(models).fit(Train_X, Train_Y)
predictions=ensemble.predict(Test_X)
submission = pd.DataFrame({ 'Application_ID': DTest['Application_ID'],'Loan_Status': predictions })
submission.to_csv("submission.csv", index=False)

