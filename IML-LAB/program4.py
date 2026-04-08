import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

df = pd.read_csv('/content/diabetes.csv')

df.head()

df.tail()

df.info()

scaler = MinMaxScaler()
df['Glucose'] = scaler.fit_transform(df[['Glucose']])
df['BloodPressure'] = scaler.fit_transform(df[['BloodPressure']])
df['SkinThickness'] = scaler.fit_transform(df[['SkinThickness']])
df['BMI'] = scaler.fit_transform(df[['BMI']])
df['Age'] = scaler.fit_transform(df[['Age']])
df['Insulin'] = scaler.fit_transform(df[['Insulin']])

df

x = df.drop('Outcome', axis=1)
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

