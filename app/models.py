from app import db
from flask_login import UserMixin
import pandas as pd 
from datetime import datetime 
from zoneinfo import ZoneInfo
from io import StringIO
from sqlalchemy import and_, not_

class Organization(db.Model):
    __tablename__ = "organization"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    dirpath = db.Column(db.String(80))
    is_dropbox = db.Column(db.Boolean, default=True)
    is_init = db.Column(db.Boolean, default=False)
    init_status = db.Column(db.String(120), default= "Please initialize the inventory" )
    init_details = db.Column(db.String(None) )
    data = db.Column(db.PickleType())

class User(UserMixin, db.Model):
    __tablename__ = "user"
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    org_id = db.Column(db.Integer, db.ForeignKey('organization.id'), nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    pwd = db.Column(db.String(300), nullable=False)
    
    # Define the relationship
    organization = db.relationship('Organization', backref=db.backref('users', lazy=True))

    def __repr__(self):
        return '<User %r>' % self.username
    
# session id is a string token
class SessionData(db.Model):
    __tablename__ = "session_data"
    
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
    __tablename__ = "tables"
    
    id = db.Column(db.Integer, primary_key=True)
    vendor = db.Column(db.String(300), unique=False, nullable=False)
    file_name = db.Column(db.String(300), unique=False, nullable=False)
    file_last_modified = db.Column(db.DateTime, nullable=False)
    file_dropbox_url = db.Column(db.String(2048), nullable=True, default=None)
    table_name = db.Column(db.String(300), unique=True, nullable=False)
    created_date = db.Column(db.DateTime, nullable=False, default=datetime.now(ZoneInfo("UTC")))
    updated_date = db.Column(db.DateTime, nullable=False, default=datetime.now(ZoneInfo("UTC")), onupdate=datetime.now(ZoneInfo("UTC")))

    @classmethod
    def get_all_sorted(cls):
        """
        Retrieves all entries from the database, sorted first by vendor in ascending order, 
        then by file_last_modified in descending order, and finally by table_name in ascending order.

        This method is useful for getting an organized list of all table entries, 
        making it easier to understand the chronological order of modifications across different vendors.

        Returns:
        - A list of Tables instances representing all entries in the database, sorted as described.
        """
        return cls.query.order_by(cls.vendor, cls.file_last_modified.desc(), cls.table_name).all()
    
    @classmethod
    def get_unique_vendors(cls):
        """
        Returns a unique list of all vendors.
        
        Returns:
        - A list of unique vendor names present in the Tables entries.
        """
        vendors = cls.query.with_entities(cls.vendor).distinct().all()
        # Extracting vendor names from the tuples returned by `.all()`
        return [vendor[0] for vendor in vendors]
    
    @classmethod
    def get_all_by_vendor(cls, vendor_name, cutoff_date:datetime = None  ):
        """
        Retrieves all entries for a specific vendor from the database, 
        ordered by the file_last_modified date in descending order, and then by file_name in ascending order.

        This method allows for a detailed view into the modifications and updates 
        made by a specific vendor, sorted to highlight the most recent changes first.

        Parameters:
        - vendor_name: A string representing the name of the vendor for which entries are to be retrieved.

        Returns:
        - A list of Tables instances representing all entries associated with the specified vendor, 
        sorted by the most recent file_last_modified dates and then by file_name.
        """
        all_files = cls.query.filter_by(vendor=vendor_name).order_by(cls.file_last_modified.desc(), cls.file_name).all()
        # remove optional last modified before 
        good_files = [f for f in all_files if f.file_last_modified > cutoff_date]

        return good_files
    
    @classmethod
    def get_most_recent_by_vendor(cls, vendor_name, str_token: str = None, ):
        """
        Returns the table entries with the most recent file_last_modified date for a given vendor:
        - one with an optional user-specified string token in the file_name
        - one without the string token in the file_name.
        
        Parameters:
        - vendor_name: A string representing the name of the vendor.
        - str_token: An optional string token to search in the file_name.
        
        Returns:
        - A tuple of two instances of the Tables class:
            - The first element is the most recent table entry for the given vendor with the str_token in the file_name.
            - The second element is the most recent table entry for the given vendor without the str_token in the file_name.
        Each element can be None if no matching entry exists.
        """
        query_base = cls.query.filter_by(vendor=vendor_name)
        
        if str_token:
            with_token = query_base.filter(cls.file_name.contains(str_token)).order_by(cls.file_last_modified.desc()).first()
        else:
            with_token = None
        
        without_token = query_base.filter(
            not_(cls.file_name.contains(str_token)) if str_token else cls.file_last_modified.isnot(None)
        ).order_by(cls.file_last_modified.desc()).first()
        
        return with_token, without_token
        
        @classmethod
        def get_unique_vendors(cls):
            """
            Returns a unique list of all vendors.
            
            Returns:
            - A list of unique vendor names present in the Tables entries.
            """
            vendors = cls.query.with_entities(cls.vendor).distinct().all()
            # Extracting vendor names from the tuples returned by `.all()`
            return [vendor[0] for vendor in vendors]

# The data saved as a python dataframe - indendent of df.columns 
class TableData(db.Model):
    __tablename__ = "table_data"
    
    table_name = db.Column(db.String(300), primary_key=True, unique=True, nullable=False)
    table_columns = db.Column(db.JSON(), nullable=True)  # Field to store a Python list as JSONB
    search_columns = db.Column(db.JSON(), nullable=True)  # Field to store a Python list as JSONB
    row_count = db.Column(db.Integer, unique=False, nullable=True)
    df = db.Column(db.JSON(), unique=False, nullable=True)
    created_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @classmethod
    def get_dataframe(cls, table_name):
        """
        Retrieves the dataframe stored as a JSON string in the database for the given table name.

        Parameters:
        - table_name: A string representing the name of the table for which the dataframe is retrieved.

        Returns:
        - A pandas DataFrame object constructed from the JSON string stored in the database if the table
        exists and contains dataframe data; otherwise, returns None.

        Raises:
        - ValueError: If the `table_name` is not a string or if it is empty.
        """
        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError("table_name must be a non-empty string")

        record = cls.query.filter_by(table_name=table_name).first()
        if record and record.df:
            try:
                # Convert the stored JSON string back to a pandas DataFrame
                df = pd.read_json(StringIO(record.df), orient='split')
                return df
            except Exception as e:
                raise ValueError(f"Error converting JSON to DataFrame: {e}")
        else:
            return None
    
    @classmethod
    def get_search_columns(cls, table_name):
        """
        Retrieves the list of search columns for the specified table.

        Parameters:
        - table_name: A string representing the name of the table for which the search columns are retrieved.

        Returns:
        - A list of search columns if the table exists and has defined search columns; otherwise, returns None.

        Raises:
        - ValueError: If the `table_name` is not a string or if it is empty.
        """
        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError("table_name must be a non-empty string")

        record = cls.query.filter_by(table_name=table_name).first()
        if record:
            return record.search_columns
        else:
            return None   
            
    @classmethod
    def get_table_columns(cls, table_name):
        """
        Retrieves the list of table columns for the specified table.

        Parameters:
        - table_name: A string representing the name of the table for which the columns are retrieved.

        Returns:
        - A list of table columns if the table exists ; otherwise, returns None.

        Raises:
        - ValueError: If the `table_name` is not a string or if it is empty.
        """
        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError("table_name must be a non-empty string")

        record = cls.query.filter_by(table_name=table_name).first()
        if record:
            return record.table_columns
        else:
            return None   
            
