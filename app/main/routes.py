from flask import Flask, session, render_template, request, Blueprint, flash, jsonify
from dotenv import load_dotenv
import uuid

from app import create_app, current_user, server_session, db
from app.models import User,UserData,SessionData
from app.search.search import searchData
from app.init.init import initSearchDropBox
from app.main.forms import search_form, test_form, login_form

load_dotenv()

from app import get_absolute_template_folder

bp = Blueprint('main', __name__, template_folder='templates')

@bp.route("/", methods=("GET", "POST"), strict_slashes=False)
def index():
    # template_folder_value = get_absolute_template_folder(bp)
    # x = bp.template_folder   y = bp.root_path

    # Check if session contains a unique identifier
    if 'session_id' not in session:
        # Generate a unique identifier and store it in the session
        session['session_id'] = str(uuid.uuid4())

    # Retrieve session data from the database using the unique identifier
    session_id = session['session_id']
    session_data = SessionData.query.get(session_id)

    # create an instance from the form class
    form=search_form()
    
    result=None
    search_term=None
    
    if form.validate_on_submit():
        try:
            search_term = form.search_text.data
            # Call the searchData function
            is_html = True  # Set this according to your requirements
            result = searchData(search_term, is_html)
        except Exception as e:
            flash(e, "Error in search_form.search_text.data")

    return render_template("index.html", 
                           form=form,
                           title="Plantfinder Dev", 
                           template_folder='templates',
                           result=result,
                           search_term=search_term)

# called from button event via javascript 
@bp.route('/init/', methods=('GET','POST'))
def init_data():
    subDirFilter = ['Canadale']

    # Generate a unique identifier for the session data
    session_id = session['session_id']

    # initialize the search data using an archived DropBox refresh token
    init_status = initSearchDropBox(onlySubdir=subDirFilter)

    # Store the session data in the database
    session_data = SessionData(id=session_id, data={'init_status': init_status["status"],
                                                    'filesfound': init_status['filesfound'],
                                                    'filesloaded': init_status['filesloaded']})
    db.session.merge(session_data)
    db.session.commit()

    temp = jsonify(init_status=init_status)
    temp2 = init_status['status']
    return temp

@bp.route('/store_data')
def store_data():
    # Generate a unique identifier for the session data
    session_id = server_session['session_id']

    # Store the session data in the database
    session_data = SessionData(id=session_id, data={'example_key': 'example_value'})
    db.session.merge(session_data)
    db.session.commit()

    return 'Data stored in the database.'

