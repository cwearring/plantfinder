from flask import Blueprint

bp = Blueprint('search', __name__,
               template_folder='templates',
               static_folder='static')

from app.search import search 
from app.search import routes
