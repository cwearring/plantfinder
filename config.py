# used in create_app instead of config statements 
from datetime import timedelta

'''import os
from dotenv import load_dotenv
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))'''

class Config:
    SECRET_KEY = 'your_very_secret_key_here'
    SESSION_TYPE = 'sqlalchemy'
    SESSION_SQLALCHEMY_TABLE = 'sessions'
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    # Use these to persist a user even when the browser is closed
    PERMANENT_SESSION_LIFETIME = timedelta(days=1)
    REMEMBER_COOKIE_DURATION = timedelta(days=1)
    # Set these to true for debugging and remove or set to False for production
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    EXPLAIN_TEMPLATE_LOADING = False
    TESTING = False
    # Default database URI, override in subclasses as needed
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///default.db'
    # docker run --name mypostgres --network mynetwork -e POSTGRES_PASSWORD=cwearring -p 5432:5432 -d postgres
    # SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://postgres:cwearring@localhost:5432/postgres'

    # Connection pool settings
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'max_overflow': 5,
        'pool_timeout': 30,
        'pool_recycle': 1800,
        #'echo_pool': 'debug', # uncomment to get verbose logging messages 
        'pool_pre_ping': True,
    }

class DockerLocalConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://postgres:cwearring@localhost:5432/postgres'

class AwsDevConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://cwearring:pa$$w0rd@pg15-2.chieuysau35f.us-east-1.rds.amazonaws.com:5432/dev1'

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://postgres:cwearring@localhost:5432/postgres'

class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://postgres:password@localhost/prod_db'
    # Consider enabling in production
    SESSION_COOKIE_SECURE = True

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///memory'

'''class HiddenForMail:
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 25)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS') is not None
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    ADMINS = ['your-email@example.com']'''

