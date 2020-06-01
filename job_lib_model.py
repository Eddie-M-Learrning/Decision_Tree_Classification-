from sklearn.externals import joblib
ks = joblib.load('salary_prob')

print(ks.predict([[1,0,1]]))