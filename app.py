from flask import Flask, render_template, request, redirect
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # nama = request.form['nama']
        bmi = float(request.form['bmi'])
        smoking = request.form['smoking']
        aldrink = request.form['aldrink']
        stroke = request.form['stroke']
        physicalh = float(request.form['physicalh'])
        mentalh = float(request.form['mentalh'])
        diffwl = request.form['diffwl']
        sex = request.form['sex']
        age = request.form['age']
        race = request.form['race']
        diabetic = request.form['diabetic']
        pactivity = request.form['pactivity']
        genh = request.form['genh']
        sleept = float(request.form['sleept'])
        asthma = request.form['asthma']
        kidneyd = request.form['kidneyd']
        skinc = request.form['skinc']

        # ML #
        data = pd.read_csv("csv/heart_2020_cleaned.csv", delimiter=",")

        data.columns = ['Y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',
                        'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17']

        X = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
                  'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17']].values

        from sklearn import preprocessing
        con_x2 = preprocessing.LabelEncoder()
        con_x2.fit(['Yes', 'No'])
        X[:, 1] = con_x2.transform(X[:, 1])

        con_x3 = preprocessing.LabelEncoder()
        con_x3.fit(['Yes', 'No'])
        X[:, 2] = con_x3.transform(X[:, 2])

        con_x4 = preprocessing.LabelEncoder()
        con_x4.fit(['Yes', 'No'])
        X[:, 3] = con_x4.transform(X[:, 3])

        con_x7 = preprocessing.LabelEncoder()
        con_x7.fit(['Yes', 'No'])
        X[:, 6] = con_x7.transform(X[:, 6])

        con_x8 = preprocessing.LabelEncoder()
        con_x8.fit(['Female', 'Male'])
        X[:, 7] = con_x8.transform(X[:, 7])

        con_x9 = preprocessing.LabelEncoder()
        con_x9.fit(['18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                   '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'])
        X[:, 8] = con_x9.transform(X[:, 8])

        con_x10 = preprocessing.LabelEncoder()
        con_x10.fit(['White', 'Black', 'Hispanic',
                    'American Indian/Alaskan Native', 'Asian',  'Other', ])
        X[:, 9] = con_x10.transform(X[:, 9])

        con_x11 = preprocessing.LabelEncoder()
        con_x11.fit(['Yes', 'No', 'Yes (during pregnancy)',
                    'No, borderline diabetes'])
        X[:, 10] = con_x11.transform(X[:, 10])

        con_x12 = preprocessing.LabelEncoder()
        con_x12.fit(['Yes', 'No'])
        X[:, 11] = con_x3.transform(X[:, 11])

        con_x13 = preprocessing.LabelEncoder()
        con_x13.fit(['Very good', 'Fair', 'Good', 'Excellent', 'Poor'])
        X[:, 12] = con_x13.transform(X[:, 12])

        con_x15 = preprocessing.LabelEncoder()
        con_x15.fit(['Yes', 'No'])
        X[:, 14] = con_x3.transform(X[:, 14])

        con_x16 = preprocessing.LabelEncoder()
        con_x16.fit(['Yes', 'No'])
        X[:, 15] = con_x3.transform(X[:, 15])

        con_x17 = preprocessing.LabelEncoder()
        con_x17.fit(['Yes', 'No'])
        X[:, 16] = con_x17.transform(X[:, 16])

        y = data[['Y']].values

        from sklearn.model_selection import train_test_split
        X_trainset, X_testset, y_trainset, y_testset = train_test_split(
            X, y, test_size=0.2, random_state=3)

        # model Devition Tree
        heartDiseaseTree = DecisionTreeClassifier(
            criterion="gini", max_depth=10)
        heartDiseaseTree.fit(X_trainset, y_trainset)

        # Prediksi
        x_new = np.array((bmi, smoking, aldrink, stroke, physicalh, mentalh, diffwl,
                         sex, age, race, diabetic, pactivity, genh, sleept, asthma, kidneyd, skinc))
        x_new = np.reshape(x_new, (1, -1))

        predTree = heartDiseaseTree.predict(x_new)

        output = predTree[0]

        return render_template('index.html', output=output)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
