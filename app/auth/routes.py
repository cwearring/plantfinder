from flask import Flask, render_template,redirect,flash,url_for,session,Blueprint,current_app
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import UserMixin,login_user,LoginManager, current_user,logout_user,login_required
from sqlalchemy.exc import IntegrityError,DataError,DatabaseError,InterfaceError,InvalidRequestError
from werkzeug.routing import BuildError
from datetime import timedelta

from app import login_manager
from app import bcrypt
from app.auth import bp 
from app.auth.forms import LoginForm, RegisterForm
from app.models import User, Organization, db

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@bp.before_request
def session_handler():
    session.permanent = True
    current_app.permanent_session_lifetime = timedelta(minutes=10)

@bp.route("/login/", methods=("GET", "POST"), strict_slashes=False)
def login():
    form = LoginForm()
    form.org.choices = [(c.id, c.name) for c in Organization.query.all()]

    if form.validate_on_submit():
        try:
            org = form.org.data 
            user = User.query.filter_by(email=form.email.data).first()
            if bcrypt.check_password_hash(user.pwd, form.pwd.data):
                login_user(user)
                return redirect(url_for('main.index'))
            else:
                flash("Invalid Username or password!", "danger")
        except Exception as e:
            flash(e, "danger")

    return render_template("auth.html",
        form=form,
        text="Login",
        title="Login to Plantfinder",
        btn_action="Login"
        )

# Register route
@bp.route("/register/", methods=("GET", "POST"), strict_slashes=False)
def register():
    form = RegisterForm()
    form.org.choices = [(c.id, c.name) for c in Organization.query.all()]

    if form.validate_on_submit():
        try:
            org = form.org.data 
            email = form.email.data
            pwd = form.pwd.data
            username = form.username.data

            newuser = User(
                org_id = org,
                username=username,
                email=email,
                pwd=bcrypt.generate_password_hash(pwd).decode('utf-8'),
            )
    
            db.session.add(newuser)
            db.session.commit()
            flash(f"Account Succesfully created", "success")
            return redirect(url_for("auth.login"))

        except InvalidRequestError:
            db.session.rollback()
            flash(f"Something went wrong!", "danger")
        except IntegrityError:
            db.session.rollback()
            flash(f"User already exists!.", "warning")
        except DataError:
            db.session.rollback()
            flash(f"Invalid Entry", "warning")
        except InterfaceError:
            db.session.rollback()
            flash(f"Error connecting to the database", "danger")
        except DatabaseError:
            db.session.rollback()
            flash(f"Error connecting to the database", "danger")
        except BuildError:
            db.session.rollback()
            flash(f"An error occured !", "danger")
    return render_template("auth.html",
        form=form,
        text="Create account",
        title="Register for Plantfinder",
        btn_action="Register account"
        )

@bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))
