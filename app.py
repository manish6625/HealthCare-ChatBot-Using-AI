from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import csv

app = Flask(__name__)

# Load data and models (same as your script)
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

reduced_data = training.groupby(training['prognosis']).max()
le = preprocessing.LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier().fit(x_train, y_train)
model = SVC().fit(x_train, y_train)

# Load dictionaries
def load_description():
    description_list = {}
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]
    return description_list

def load_severity():
    severityDictionary = {}
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            severityDictionary[row[0]] = int(row[1])
    return severityDictionary

def load_precaution():
    precautionDictionary = {}
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]
    return precautionDictionary

description_list = load_description()
severityDictionary = load_severity()
precautionDictionary = load_precaution()

symptoms_dict = {symptom: idx for idx, symptom in enumerate(cols)}

# Helper functions
def predict_disease(symptoms, days):
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    # Use DataFrame with feature names to avoid warning
    input_df = pd.DataFrame([input_vector], columns=list(symptoms_dict.keys()))
    pred = clf.predict(input_df)[0]
    disease = le.inverse_transform([pred])[0]
    description = description_list.get(disease, "No description available.")
    precautions = precautionDictionary.get(disease, [])
    severity = sum(severityDictionary.get(item, 0) for item in symptoms)
    risk = (severity * days) / (len(symptoms) + 1)
    consult = risk > 13
    return disease, description, precautions, consult, severity

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    symptoms_list = list(cols)
    if request.method == 'POST':
        name = request.form.get('name')
        symptom = request.form.get('symptoms')
        symptoms = [symptom.strip().replace(' ', '_')] if symptom and symptom.strip() else []
        try:
            days = int(request.form.get('days', '1'))
        except ValueError:
            days = 1
        if symptom == 'none':
            result = {
                'name': name,
                'disease': 'None',
                'description': 'None',
                'precautions': ['None'],
                'consult': False,
                'severity': 0
            }
        else:
            disease, description, precautions, consult, severity = predict_disease(symptoms, days)
            result = {
                'name': name,
                'disease': disease,
                'description': description,
                'precautions': precautions,
                'consult': consult,
                'severity': severity
            }
    else:
        result = None
    return render_template('index.html', result=result, symptoms_list=symptoms_list)

if __name__ == '__main__':
    app.run(debug=True)
