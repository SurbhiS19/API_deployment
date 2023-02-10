import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

housing_data = pd.read_csv('USA_Housing.csv')

X= housing_data.drop(['Price','Address'],axis=1)
Y=housing_data['Price']

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.25,random_state=100)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred=lr.predict(X_train)

train_r2 = r2_score(y_train, y_pred)
print('train r2 is',train_r2)

y_pred=lr.predict(X_test)

test_r2 = r2_score(y_test, y_pred)
print('test r2 is',test_r2)

pickle.dump(lr, open('model.pkl','wb'))