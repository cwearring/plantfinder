from flask import Blueprint

bp = Blueprint('init', __name__,template_folder='templates')

from app.init import routes
from app.init import forms
from app.init import extract_table
