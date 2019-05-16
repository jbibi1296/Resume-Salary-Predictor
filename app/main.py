#!/usr/bin/env python
from flask import Flask, flash, redirect, render_template, request, url_for
from salary import run_tester
from salary import get_table
import textract
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'docx'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')

def index():
    return render_template(
        'salary.html',
        data=[{'name':'Linear Regression'}, {'name':'Lasso'}, {'name':'Ridge'},
        {'name':'Random Forest'}, {'name':'Gradient Boost'}, {'name':'Neural Network'},
        {'name':'Linear Regression Poly'}, {'name':'Lasso Poly'}, {'name':'Ridge Poly'},
        {'name':'Random Forest Poly'}, {'name':'Gradient Boost Poly'},
              {'name':'Neural Network Poly'}])
@app.route("/result" , methods=['GET', 'POST'])
def result():
    error = None
    text = request.form.get('text')
    if 'file' not in request.files:
        if text == '':
            return redirect(request.url)
        else:
            text = request.form.get('text')
    else:
        file = request.files['file']
        if file.filename == '':
            if request.form.get('text') == '':
                return redirect(request.url)
        else:
            text = request.form.get('text')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            text = textract.process(filepath)
            os.remove(filepath)
    model = request.form.get('model_select')
    text = str(text).lower()
    resp = run_tester(text,model)
    data = resp
    table = get_table(text)
    table = table.replace('<table border="1" class="dataframe">','<table border="1" class="dataframe" align="center">')
    return render_template(
        'result.html',
        data=data,
        error=error,
        table = table)

if __name__=='__main__':
    app.run(debug=False)