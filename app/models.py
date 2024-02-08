from app import db
from flask_login import UserMixin
import pandas as pd 
from datetime import datetime 
from io import StringIO

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
    file_last_modified = db.Column(db.DateTime, nullable=False)
    table_name = db.Column(db.String(300), unique=True, nullable=False)
    created_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

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
    def get_all_by_vendor(cls, vendor_name):
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
        return cls.query.filter_by(vendor=vendor_name).order_by(cls.file_last_modified.desc(), cls.file_name).all()    

    @classmethod
    def get_most_recent_by_vendor(cls, vendor_name):
        """
        Returns the table entry with the most recent file_last_modified date for a given vendor.
        
        Parameters:
        - vendor_name: A string representing the name of the vendor.
        
        Returns:
        - An instance of the Tables class representing the most recent table entry for the given vendor,
        or None if the vendor does not exist.
        """
        return cls.query.filter_by(vendor=vendor_name).order_by(cls.file_last_modified.desc()).first()
    
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
    table_name = db.Column(db.String(300), primary_key=True, unique=True, nullable=False)
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
            
