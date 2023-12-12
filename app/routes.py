from flask import (
    Flask,
    render_template,
    Blueprint
)

bp = Blueprint('main', __name__)

@bp.route("/", methods=("GET", "POST"), strict_slashes=False)
def index():
    return render_template("index.html",title="Home")

