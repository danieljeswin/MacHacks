from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, BooleanField, SelectField, IntegerField, DateField, FileField
from wtforms.validators import ValidationError, DataRequired, Length, NumberRange



class AddVideoForm(FlaskForm):
    video = FileField('Video File', validators=[DataRequired()])
    name = StringField('Video Name', validators=[DataRequired()])
    classname = SelectField('Class Name', validators=[DataRequired()])
    submit = SubmitField('Add')
    
    