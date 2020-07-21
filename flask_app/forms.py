from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField, PasswordField
from wtforms.validators import DataRequired, EqualTo, Required, InputRequired, Length, Optional


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[Length(min=4, max=25)])
    email = StringField('Email Address', validators=[Length(min=6, max=35)])
    password = StringField('Password', validators=[DataRequired()])
    confirm_password = StringField('Confirm password', validators=[
        EqualTo('password', message='Password and Confirm password do not match'), DataRequired()])
    accept_terms = BooleanField('Accept Terms', validators=[InputRequired()])


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = StringField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me', validators=[Optional()])
