from wtforms import StringField,PasswordField,BooleanField,IntegerField, DateField,TextAreaField,ValidationError,validators
from flask_wtf import FlaskForm
from wtforms.validators import InputRequired, Length, EqualTo, Email, Regexp ,Optional
from flask_login import current_user
from wtforms import ValidationError,validators

class init_form(FlaskForm):
    init_dirpath = StringField(
        validators=[Optional()]
    )
