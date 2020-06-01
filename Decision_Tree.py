import pandas as pd 


df =pd.read_csv("Salary_company.csv")
print(df.head())

inputs = df.drop("salary" , axis = 'columns')
targets = df['salary']
print(inputs)
print(targets)

from sklearn.preprocessing import LabelEncoder
le =  LabelEncoder()

inputs['company']  = le.fit_transform(inputs['company'])
inputs['degree']  = le.fit_transform(inputs['degree'])
inputs['job']  = le.fit_transform(inputs['job'])
print(inputs)



# = inputs.drop(['company','job','degree'],axis = 'columns')
#print(input_n)

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs , targets)
print(model.score(inputs , targets))

print(model.predict([[1,4,1]]))

from sklearn.externals import joblib
joblib.dump(model , 'salary_prob')
mj = joblib.load('salary_prob')
print(mj.predict([[1,4,1]]))

