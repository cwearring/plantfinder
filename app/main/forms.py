from wtforms import StringField,PasswordField,BooleanField,IntegerField, DateField,TextAreaField,ValidationError,validators
from flask_wtf import FlaskForm
from wtforms.validators import InputRequired, Length, EqualTo, Email, Regexp ,Optional
import email_validator
from wtforms import ValidationError,validators
from app.models import User, SessionData

class SearchForm(FlaskForm):
    search_text = StringField(
        validators=[Optional()]
    )


