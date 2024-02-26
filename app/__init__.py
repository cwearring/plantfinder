# this file is run when the package is loaded 
from flask import Flask
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from config import DevelopmentConfig, DockerLocalConfig, AwsDevConfig, ProductionConfig, Config

from flask_migrate import Migrate
from flask_login import LoginManager
from datetime import timedelta
# for realtime updates to a section of the page - init progress feedback 
# https://pypi.org/project/eventlet/

db = SQLAlchemy()
migrate = Migrate()
bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.session_protection = "strong"
login_manager.login_view = "auth.login"
login_manager.login_message_category = "info"

import logging
from logging.handlers import RotatingFileHandler
import os 
from dotenv import load_dotenv
load_dotenv()
my_config = os.getenv('CONFIG_LEVEL', None)

print(f"my_config __init__ = {my_config} - read in __init__ top level ")

# Function to get the absolute template folder path - for debugging
# def get_absolute_template_folder(bp):     return bp.jinja_loader.searchpath[0] if bp.jinja_loader.searchpath else None

# gunicorn -b 127.0.0.1:5000 app:myapp - 2024-02-08
def setup_logging(app):
        log_level = logging.INFO
        log_file = './logfiles/application.log'
        file_handler = RotatingFileHandler(log_file, maxBytes=10240, backupCount=10)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler)

def create_app(config_name=None):
    app = Flask(__name__)
    # move this config inside the app factory function 
    my_config = os.getenv('CONFIG_LEVEL', None)
    print(f"config_name = {config_name} - read inside create_app - load_dotenv outside create_app()")
      
    if config_name == 'dev':
        app.config.from_object(DevelopmentConfig)
    elif config_name == 'docker_local':
        app.config.from_object(DockerLocalConfig)
    elif config_name == 'aws_dev1':  # test postgres db 'dev1'- public IG port 5432
        app.config.from_object(AwsDevConfig)    
    elif config_name == 'prod':
        app.config.from_object(ProductionConfig)
    else:
        app.config.from_object(Config)

    # Set up logging
    if not app.debug:
        setup_logging(app)
 
    # Set Werkzeug logger level to WARNING
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)

    # Extensions Initialization
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    bcrypt.init_app(app)
    
    # Blueprint registration
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    from app.init import bp as init_bp
    app.register_blueprint(init_bp, url_prefix='/init')

    from app.search import bp as search_bp
    app.register_blueprint(search_bp, url_prefix='/search')

    from app.main.routes import bp as main_bp
    app.register_blueprint(main_bp, url_prefix='')

    return app

myapp=create_app(config_name = my_config)
