from app import db
from flask_login import UserMixin
import pandas as pd 
from datetime import datetime 

class User(UserMixin, db.Model):
    __tablename__ = "user"
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    pwd = db.Column(db.String(300), nullable=False, unique=True)

    def __repr__(self):
        return '<User %r>' % self.username
    
# user id is an integer 
class UserData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.PickleType())

    @classmethod
    def get_user_data(cls, user_id):
        user_data = cls.query.get(user_id)
        if user_data:
            return user_data.data
        else:
            return None

# session id is a string token
class SessionData(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    data = db.Column(db.PickleType())

# Track the status of long running initialize background thread 
class ThreadComplete(db.Model):
    __tablename__ = 'threads'
    id = db.Column(db.Integer, primary_key=True)
    task_complete = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"<ThreadComplete(id='{self.id}', task_complete={self.task_complete})>"

    @classmethod
    def is_task_complete(cls, id):
        record = cls.query.filter_by(id=id).first()
        return record.task_complete if record else None
    
# A list of all the files we have processed into tables 
class Tables(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    vendor = db.Column(db.String(300), unique=False, nullable=False)
    file_name = db.Column(db.String(300), unique=True, nullable=False)
    table_name = db.Column(db.String(300), unique=True, nullable=False)
    created_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    @classmethod
    def get_all_sorted(cls):
        return cls.query.order_by(cls.vendor, cls.updated_date.desc(), cls.table_name).all()
    
    @classmethod
    def get_all_by_vendor(cls, vendor_name):
        # This orders the results for a specific vendor by updated_date (desc) and then by file_name
        return cls.query.filter_by(vendor=vendor_name).order_by(cls.updated_date.desc(), cls.file_name).all()    

# The data saved as a python dataframe - indendent of df.columns 
class TableData(db.Model):
    table_name = db.Column(db.String(300), primary_key=True, unique=True, nullable=False)
    search_columns = db.Column(db.JSON(), nullable=True)  # Field to store a Python list as JSONB
    row_count = db.Column(db.Integer, unique=False, nullable=True)
    df = db.Column(db.JSON(), unique=False, nullable=True)
    created_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @classmethod
    def get_dataframe(cls, table_name):
        # Query the database for the record with the given table_name
        record = cls.query.filter_by(table_name=table_name).first()
        if record and record.df:
            # Convert the JSONB stored data back into a pandas DataFrame
            df = pd.read_json(record.df, orient='split')
            return df
        else:
            return None    

    @classmethod
    def get_search_columns(cls, table_name):
        # Method to retrieve the Python list from search_columns for a specific table
        record = cls.query.filter_by(table_name=table_name).first()
        if record:
            return record.search_columns
        else:
            return None        