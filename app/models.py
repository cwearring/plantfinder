from app import db
from flask_login import UserMixin

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

# session id is a string token
class SessionData(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    data = db.Column(db.PickleType())

# Function to retrieve user data by user id
def get_user_data(user_id):
    user_data = UserData.query.get(user_id)
    if user_data:
        return user_data.data
    else:
        return None


# Track the status of long running initialize background thread 
class ThreadComplete(db.Model):
    __tablename__ = 'threads'
    id = db.Column(db.String, primary_key=True)
    task_complete = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"<ThreadComplete(id='{self.id}', task_complete={self.task_complete})>"

    @classmethod
    def is_task_complete(cls, id):
        record = cls.query.filter_by(id=id).first()
        return record.task_complete if record else None
    

