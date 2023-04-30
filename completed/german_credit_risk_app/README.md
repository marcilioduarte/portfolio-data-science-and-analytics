This is a case study about creditworthiness classification where I did the whole process of XXX, from the data cleansing part until the deployment of the model in an application simulated for the banks managers. To achieve this goal, I analyzed and prepared the dataset for machine learning models. The applied models are: Logistic Regression, Decision Tree Classifier, and Random Forest Classifier, which are available in Python's sklearn library. To optimize the workflow and model results, I applied a personalized pipeline for model application and GridSearchCV for parameter optimization. The app development was made using gradio app.

The data is uploaded in path: german_credit_risk/data/raw, but it was first obtained from Kaggle  and can be obtained [HERE](https://www.kaggle.com/datasets/mpwolke/cusersmarildownloadsgermancsv).

I used a previous work from Pennsylvania State University as a reference in many parts of the code, you can find it [HERE] (https://online.stat.psu.edu/stat508/resource/analysis/gcd). Also, as this is a case study, the code steps are commented in english.

It is worth mentioning that the feature selection process was carefully performed according to the regulations of the Central Bank of Brazil, as I'm Brazilian.

---

title: Credit Worthiness Risk Classification

sdk: gradio

sdk_version: 3.27.0

app_file: app.py

license: apache-2.0

---
