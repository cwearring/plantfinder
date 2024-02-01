# this file is run when the package is loaded 
from flask import Flask, render_template,redirect,flash,url_for,session,Blueprint
from flask_bcrypt import Bcrypt,generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import UserMixin,login_user,LoginManager,current_user,logout_user,login_required
from sqlalchemy.exc import IntegrityError,DataError,DatabaseError,InterfaceError
from werkzeug.routing import BuildError
from datetime import timedelta
# to save session data - db models defined in models/.py
from flask_session import Session
# for realtime updates to a section of the page - init progress feedback 
from flask_socketio import SocketIO
# https://pypi.org/project/eventlet/

db = SQLAlchemy()
migrate = Migrate()
bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.session_protection = "strong"
login_manager.login_view = "auth.login"
login_manager.login_message_category = "info"

# Function to get the absolute template folder path - for debugging
def get_absolute_template_folder(bp):
    return bp.jinja_loader.searchpath[0] if bp.jinja_loader.searchpath else None

def create_app():
    app = Flask(__name__)

    # Flask-Session Configuration
    app.secret_key = 'secret-key'
    app.config['SESSION_TYPE'] = 'sqlalchemy'
    app.config['SESSION_SQLALCHEMY'] = db  # Use the same SQLAlchemy instance as your app
    app.config['SESSION_SQLALCHEMY_TABLE'] = 'sessions'
    app.config["SESSION_PERMANENT"] = False
    app.config['SESSION_USE_SIGNER'] = True

    # use these to persist a user even when the browser is closed 
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1) 
    app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=1)

    # set these to true for debugging and remove for production
    app.config['EXPLAIN_TEMPLATE_LOADING'] = False
    app.config['TESTING'] = True

    # Flask-SQLAlchemy and DataStore Configuration
    # app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///database.db"

    # local docker postgres image - no vector store as of 2024-01-23
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://postgres:cwearring@localhost:5432/postgres'

    # Set SESSION_COOKIE_SECURE to True in a production environment with HTTPS
    # app.config['SESSION_COOKIE_SECURE']

    # Extensions Initialization
    login_manager.init_app(app)
    db.init_app(app)
    migrate.init_app(app, db)
    bcrypt.init_app(app)
    # server_session = Session()

    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    from app.init import bp as init_bp
    app.register_blueprint(init_bp, url_prefix='/init')

    from app.search import bp as search_bp
    app.register_blueprint(search_bp, url_prefix='/search')
    
    from app.main.routes import bp as main_bp
    app.register_blueprint(main_bp, url_prefix='')

    x0 = get_absolute_template_folder(main_bp)
    x1 = get_absolute_template_folder(auth_bp)
    x2 = get_absolute_template_folder(init_bp)
    x3 = get_absolute_template_folder(search_bp)
    jnk=0

    return app


def create_app_planttableextract():
    app = Flask(__name__)

    # Flask-Session Configuration
    # app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_TYPE'] = 'sqlalchemy'
    app.config['SESSION_SQLALCHEMY'] = db  
    app.config['SESSION_SQLALCHEMY_TABLE'] = 'sessions'
    app.config["SESSION_PERMANENT"] = True
    app.config['SESSION_USE_SIGNER'] = True
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1) 
    app.config['SESSION_COOKIE_SECURE'] = True
    app.secret_key = 'secret-key'

    # Flask-SQLAlchemy and DataStore Configuration
    # local docker postgres image - no vector store as of 2024-01-23
    SQLALCHEMY_DATABASE_URI ='postgresql+psycopg2://postgres:cwearring@localhost:5432/postgres'
    app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI

    # Other Flask App Configurations
    app.config['EXPLAIN_TEMPLATE_LOADING'] = True
    app.config['TESTING'] = True
    
    from . import routes
    # from .routes import bp
    # app.register_blueprint(bp)

    # Extensions Initialization
    db.init_app(app)
    with app.app_context():
        db.create_all()
    migrate.init_app(app, db)

    return app
