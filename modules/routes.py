from flask import render_template, redirect, Response, url_for
from modules import app
from modules.forms import PastYearForm, NewsForm
from phase1 import predict
from phase2 import predict_news

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    form = PastYearForm()
    form1 = NewsForm()
    if form.validate_on_submit() and form.submit.data:
        company = form.company.data
        budget = form.budget.data
        year = form.year.data
        print(company, budget, year)
        predict(company, budget, year)
        return redirect(url_for('result'))
    
    if form1.validate_on_submit() and form1.submit1.data:
        company = form1.company.data
        budget = form1.budget.data
        year = form1.year.data
        print(company, budget, year)
        data = predict_news(company, budget, year)
        return render_template('news_result.html', data = data)

    return render_template('base.html', form = form, form1 = form1)

@app.route('/result')
def result():    
    return render_template('result.html')

@app.route('/news_result')
def news_result():
    return render_template('news_result.html', data)