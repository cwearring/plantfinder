from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,SelectField #, BooleanField,IntegerField, DateField,TextAreaField
from wtforms.validators import InputRequired, Length, EqualTo, Email, Regexp ,Optional
from email_validator import validate_email, EmailNotValidError
from flask_login import current_user
from wtforms import ValidationError,validators
from app.models import User


class LoginForm(FlaskForm):
    org = SelectField('Organization', coerce=int)
    email = StringField(validators=[InputRequired(), Email(), Length(1, 64)])
    pwd = PasswordField(validators=[InputRequired(), Length(min=8, max=72)])
    username = StringField(validators=[Optional()])


class RegisterForm(FlaskForm):
    org = SelectField('Organization', coerce=int)
    username = StringField(
        validators=[
            InputRequired(),
            Length(3, 20, message="Please provide a valid name"),
            Regexp(
                "^[A-Za-z][A-Za-z0-9_.]*$",
                0,
                "Usernames must have only letters, " "numbers, dots or underscores",
            ),
        ]
    )
    email = StringField(validators=[InputRequired(), Email(), Length(1, 64)])
    pwd = PasswordField(validators=[InputRequired(), Length(8, 72)])
    cpwd = PasswordField(
                validators=[InputRequired(), Length(8, 72), 
                            EqualTo("pwd", message="Passwords must match !"),
                ] 
            )

    def validate_email(self, email):
        if User.query.filter_by(email=email.data).first():
            raise ValidationError("Email already registered!")

    def validate_uname(self, uname):
        if User.query.filter_by(username=uname.data).first():
            raise ValidationError("Username already taken!")