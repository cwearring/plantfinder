from flask import Flask, render_template,Blueprint
from app import create_app
from dotenv import load_dotenv

load_dotenv()

from app import get_absolute_template_folder

bp = Blueprint('main', __name__, template_folder='templates')

@bp.route("/", methods=("GET", "POST"), strict_slashes=False)
def index():
    template_folder_value = get_absolute_template_folder(bp)
    x = bp.template_folder
    y = bp.root_path

    return render_template("index.html", title="Home", template_folder='templates')

