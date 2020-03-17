from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired

CHOICES = [('AAPL', 'Apple'),('GOOG', 'Google'),('AMZN', 'Amazon'),('COP', 'Phillips'),('INFY', 'Infosys'),('IBM', 'IBM'),('QCOM', 'Qualcomm'),('ADBE', 'Adobe'),('WMT', 'Walmart')]

class PastYearForm(FlaskForm):
    company = SelectField(label='Company', choices=CHOICES)
    budget = StringField('Budget', validators=[DataRequired()])
    year = StringField('Year', validators=[DataRequired()])
    submit = SubmitField('Predict')


class NewsForm(FlaskForm):
    company = SelectField(label='Company', choices=CHOICES)
    budget = StringField('Budget', validators=[DataRequired()])
    year = StringField('Year', validators=[DataRequired()])
    submit1 = SubmitField('Predict')