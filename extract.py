import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import * #识别中文字体用
data=pd.read_csv('tt/train.csv')
# print(data.columns)
data=data[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]
# print(data.head())
#观察客舱等级与获救情况关系
mpl.rcParams['font.sans-serif']=['SimHei']  #画图识别中文
Survived_0=data.Pclass[data.Survived==0].value_counts()
Survived_1=data.Pclass[data.Survived==1].value_counts()
df=pd.DataFrame({u'获救':Survived_1,u'未获救':Survived_0})
df.plot(kind='bar',stacked=True,color=['lightcoral','lightgreen'])  #获救为绿色，未获救为红色
plt.title(u'各客舱等级的获救情况')
plt.xlabel(u'客舱等级')
plt.ylabel(u'人数')
plt.show()
#查看性别对获救的影响
Survived_m=data.Survived[data['Sex']=='male'].value_counts()
Survived_f=data.Survived[data['Sex']=='female'].value_counts()
df=pd.DataFrame({u'男性':Survived_m,u'女性':Survived_f})
df.plot(kind='bar',stacked=True,color=['pink','royalblue'])
plt.xlabel(u"性别")
plt.ylabel(u"人数")
plt.show()

data['Age']=data['Age'].fillna(data['Age'].median())
data['Cabin']=pd.factorize(data.Cabin)[0]
data.fillna(0,inplace=True)
data['Sex']=[1 if x=='male' else 0 for x in data.Sex]
data['p1']=np.array(data['Pclass']==1).astype(np.int32)
data['p2']=np.array(data['Pclass']==2).astype(np.int32)
data['p3']=np.array(data['Pclass']==3).astype(np.int32)
del data['Pclass']
# print(data.Embarked.unique())
data['e1']=np.array(data['Embarked']=='S').astype(np.int32)
data['e2']=np.array(data['Embarked']=='C').astype(np.int32)
data['e3']=np.array(data['Embarked']=='Q').astype(np.int32)
del data['Embarked']


print(data.describe())

data_train=data[['Sex','Age','SibSp','Parch','Fare','Cabin','p1','p2','p3','e1','e2','e3']]
data_target=data['Survived'].values.reshape(len(data),1)
print(np.shape(data_train),np.shape(data_target))


#test数据预处理
data_test=pd.read_csv('tt/test.csv')
# print(data_test.columns)
data_test=data_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]
# print(data_test.head())
data_test['Age']=data_test['Age'].fillna(data_train['Age'].median())
data_test['Cabin']=pd.factorize(data_test.Cabin)[0]
data_test.fillna(0,inplace=True)
data_test['Sex']=[1 if x=='male' else 0 for x in data_test.Sex]
data_test['p1']=np.array(data_test['Pclass']==1).astype(np.int32)
data_test['p2']=np.array(data_test['Pclass']==2).astype(np.int32)
data_test['p3']=np.array(data_test['Pclass']==3).astype(np.int32)
del data_test['Pclass']
# print(data_test.Embarked.unique())
data_test['e1']=np.array(data_test['Embarked']=='S').astype(np.int32)
data_test['e2']=np.array(data_test['Embarked']=='C').astype(np.int32)
data_test['e3']=np.array(data_test['Embarked']=='Q').astype(np.int32)
del data_test['Embarked']

#训练数据
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
#1、LogisticRegression
logreg=LogisticRegression()
logreg.fit(data_train,data_target)
Y_pred=logreg.predict(data_test)

#2、随机森林
random_forest=RandomForestClassifier(n_estimators=50)
random_forest.fit(data_train,data_target)
Y_pred_forest=random_forest.predict(data_test)

#3、决策树
clf=tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(data_train,data_target)
Y_pred_clf=clf.predict(data_test)

test_df=pd.read_csv('tt/gender_submission.csv')
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred_clf
    })
submission.to_csv('result/decisionTree.csv',index=False)


