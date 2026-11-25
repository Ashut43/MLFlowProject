import numpy as np
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
data=sns.load_dataset('diamonds')
columns1=data.columns
print(data['clarity'].unique())
data['cut']=data['cut'].astype(str)
data['cut']=data['cut'].map({
    "Fair":0,"Good":1,
    "Very Good":2,"Premium":3,"Ideal":4
})
data['clarity']=data['clarity'].astype(str)
data['clarity']=data['clarity'].replace({
    'IF':0,"VVS1":1,"VVS2":2,"VS1":3,"VS2":4,"SI1":5,"SI2":6,"I1":7
})
data['color']=data['color'].astype(str)
data['color']=data['color'].replace({
    "E":0,"I":1,"J":2,"H":3,"F":4,"G":5,"D":6
})
data['clarity']=data['clarity'].astype(float)
data['color']=data['color'].astype(np.int64)
#print(data['cut'].value_counts())
print(data.isnull().sum())
data=data.drop_duplicates()
print(data.sample(3))
print(data.columns)
print(data.info())
# plt.figure(figsize=(12,10))
# j=1
# for i in columns1:
#     plt.subplot(3,4,j)
#     sns.kdeplot(data[i])
#     plt.title(f'{i} v/s probabilityDensity')
#     j+=1
# plt.show()
x_train,x_test,y_train,y_test=train_test_split(data.drop(columns=['price']),data['price'],test_size=0.2,random_state=42)
n_estimators=100
max_depth=2
rf=RandomForestRegressor(
    n_estimators=n_estimators,
    criterion="squared_error",
    max_depth=max_depth
)
with mlflow.start_run():
    rf.fit(x_train,y_train)
    y_pred=rf.predict(x_test)
    r2_score1=r2_score(y_test,y_pred)
    mae =mean_absolute_error(y_test, y_pred)
    mse =mean_squared_error(y_test, y_pred)
    rmse =np.sqrt(mse)

    mlflow.log_metric("r2_score",r2_score1)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_param('n_estimatores',n_estimators)
    mlflow.log_param('max_depth',max_depth)

print(r2_score1)

