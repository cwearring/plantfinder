from flask import Blueprint

bp = Blueprint('init', __name__,
               template_folder='templates',
               static_folder='static')

# from app.init import routes 

