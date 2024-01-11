from flask import Blueprint

bp = Blueprint('init', __name__,template_folder='templates')

from app.init import init 
from app.init import routes