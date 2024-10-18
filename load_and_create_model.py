 # Import mediapipe
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle
import numpy as np
import os
if "body_language.pkl" in os.listdir():
    os.remove("body_language.pkl")
df=pd.read_csv("new_coords.csv")
# df = df[df['class'].isin(["alef","baA","tha","ta"])]
x = df.drop('class',axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression(n_jobs=-1,penalty="l2",C=0.1)),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier(n_jobs=-1)),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
# for algo, pipeline in pipelines.items():
#     model = pipeline.fit(X_train, y_train)
#     fit_models[algo] = model
fittedlr=pipelines["lr"].fit(X_train,y_train)
# for algo, model in fit_models.items():
#     yhat = model.predict(X_test)
#     print(algo, accuracy_score(y_test, yhat))
y_pred=fittedlr.predict(X_test)
print(accuracy_score(y_test,y_pred))
# cvsc=cross_val_score(fit_models['rf'],x,y,cv=5)
# print(cvsc.mean())
# y_predict=fit_models['rf'].predict(X_test)
# print(accuracy_score(y_test,y_predict))

# with open("result.txt","w") as f:
#     results = []
#     for result,proba in zip(fit_models['lr'].predict(X_test),model.predict_proba(X_test)):
#         results.append(result)
#         results.append(str(round(proba[np.argmax(proba)],2)))
#     f.write(str(results))

with open('body_language.pkl', 'wb') as f:
    pickle.dump(fittedlr, f)
