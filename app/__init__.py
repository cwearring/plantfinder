# this file is run when the package is loaded 
from flask import (
    Flask,
    render_template,
    redirect,
    flash,
    url_for,
    session,
    Blueprint
)
from flask_bcrypt import Bcrypt,generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import (
    UserMixin,
    login_user,
    LoginManager,
    current_user,
    logout_user,
    login_required,
)
from sqlalchemy.exc import (
    IntegrityError,
    DataError,
    DatabaseError,
    InterfaceError,
    InvalidRequestError,
)
from werkzeug.routing import BuildError
from datetime import timedelta

db = SQLAlchemy()
migrate = Migrate()
bcrypt = Bcrypt()

# added this code from app.py
login_manager = LoginManager()
login_manager.session_protection = "strong"
login_manager.login_view = "login"
login_manager.login_message_category = "info"

def create_app():
    app = Flask(__name__)

    app.secret_key = 'secret-key'
    app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///database.db"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

    login_manager.init_app(app)
    db.init_app(app)
    migrate.init_app(app, db)
    bcrypt.init_app(app)

    from app.routes import bp as main_bp
    app.register_blueprint(main_bp, url_prefix='')

    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='')

    from app.init import bp as init_bp
    app.register_blueprint(init_bp, url_prefix='/init')

    from app.search import bp as search_bp
    app.register_blueprint(search_bp, url_prefix='/search')
    
    return app
