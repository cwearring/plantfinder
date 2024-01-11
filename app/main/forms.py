from wtforms import StringField,PasswordField,BooleanField,IntegerField, DateField,TextAreaField,ValidationError,validators
from flask_wtf import FlaskForm
from wtforms.validators import InputRequired, Length, EqualTo, Email, Regexp ,Optional
import email_validator
from flask_login import current_user
from wtforms import ValidationError,validators
from app.models import User, SessionData


class search_form(FlaskForm):
    search_text = StringField(
        validators=[Optional()]
    )

class test_form(FlaskForm):
    test = PasswordField(validators=[InputRequired(), Length(min=8, max=72)])
    # Placeholder labels to enable form rendering
    test = StringField(
        validators=[Optional()]
    )

class login_form(FlaskForm):
    email = StringField(validators=[InputRequired(), Email(), Length(1, 64)])
    pwd = PasswordField(validators=[InputRequired(), Length(min=8, max=72)])
    # Placeholder labels to enable form rendering
    username = StringField(
        validators=[Optional()]
    )

