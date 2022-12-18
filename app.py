import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect

app = Flask(__name__)


@app.route('/prediction', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        method = 'Method'
        prediction = 'Prediction'
        tree_method = 'Decission Tree'
        knn_method = 'KNN'
        rlogic_method = 'Logic Regression'

        bmi = float(request.form['bmi'])
        smoking = float(request.form['smoking'])
        aldrink = float(request.form['aldrink'])
        stroke = float(request.form['stroke'])
        physicalh = float(request.form['physicalh'])
        mentalh = float(request.form['mentalh'])
        diffwl = float(request.form['diffwl'])
        sex = float(request.form['sex'])
        age = float(request.form['age'])
        race = float(request.form['race'])
        diabetic = float(request.form['diabetic'])
        pactivity = float(request.form['pactivity'])
        genh = float(request.form['genh'])
        sleept = float(request.form['sleept'])
        asthma = float(request.form['asthma'])
        kidneyd = float(request.form['kidneyd'])
        skinc = float(request.form['skinc'])

        

        x_new = np.array([bmi, smoking, aldrink, stroke, physicalh, mentalh, diffwl,
                         sex, age, race, diabetic, pactivity, genh, sleept, asthma, kidneyd, skinc])

        x_new = np.reshape(x_new, (1, -1))
        
        # Model Decitionn Tree #
        model_tree = pickle.load(open('model/tree_model.sav', 'rb'))
        pred_tree = model_tree.predict(x_new)

        # Model Decitionn KNN #
        model_knn = pickle.load(open('model/knn_model.sav', 'rb'))
        pred_knn = model_knn.predict(x_new)

        # Model Decitionn Logic #
        model_rlogic = pickle.load(open('model/reglogic.sav', 'rb'))
        pred_rlogic = model_rlogic.predict(x_new)

        tree = str(pred_tree[0])
        knn = str(pred_knn[0])
        rlogic = str(pred_rlogic[0])

        return render_template('predict.html', tree=tree, knn=knn, rlogic=rlogic, method=method, prediction=prediction, tree_method=tree_method, knn_method=knn_method, rlogic_method=rlogic_method)
    else:
        return render_template('predict.html')


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
