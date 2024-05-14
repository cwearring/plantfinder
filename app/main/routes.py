from flask import Flask, session, render_template,  Blueprint, flash, jsonify
from flask_login import current_user
from dotenv import load_dotenv
import uuid
import os

from app import create_app, db
from app.models import User, SessionData, Organization
from app.search.search import searchData
from app.main.forms import SearchForm

load_dotenv()
dbx_directory = os.getenv('DROPBOX_DIRECTORY')

# from app import get_absolute_template_folder

bp = Blueprint('main', __name__, template_folder='templates')

@bp.route('/health', methods=['GET'])
def health_check():
    # Perform internal health checks here (e.g., database connectivity)
    # For simplicity, we're just returning a "healthy" status
    return jsonify({"status": "healthy"}), 200

@bp.route('/details/<int:org_id>', methods=['GET'])
def get_details(org_id):
    user_org = Organization.query.get_or_404(org_id)
    return jsonify({"init_details": user_org.init_details})


@bp.route("/", methods=("GET", "POST"), strict_slashes=False)
def index():
    # template_folder_value = get_absolute_template_folder(bp)
    # x = bp.template_folder   y = bp.root_path

    # Check if session contains a unique identifier
    if 'session_id' not in session:
        # Generate a unique identifier and store it in the session
        session['session_id'] = str(uuid.uuid4())

    # check for previously existing user and org data  
    if current_user.is_authenticated:
        user_org = Organization.query.get(current_user.org_id)
    else:
        # check for existing user info 
        user_org = None

    if user_org:
        initStatusElement = user_org.init_status
    elif current_user.is_authenticated: # but no user_org matched- create a new org 
        new_org = Organization(
            name="Woodland",
            dirpath=dbx_directory,
            is_dropbox=True,  # This will use the default if not specified
            is_init=False,  # This will use the default if not specified
            init_status="Please initialize the inventory",  # This will use the default if not specified
            init_details=None,  # set the 'data' field to None initially
            data=None  # Assuming you want to set the 'data' field to None initially
        )
        initStatusElement = new_org.init_status
        
        db.session.add(new_org)
        db.session.commit()

    else:
        initStatusElement = "Please initialize the inventory"  

    # create an instance from the form class
    form=SearchForm()
    
    result=None
    search_term=None
    result_tables = None
    result_url = None               
    
    if form.validate_on_submit():
        try:
            search_term = form.search_text.data
            # Call the searchData function
            is_html = True  # Set this according to your requirements

            # returns a tuple (theOutput, theUrls)
            result = searchData(search_term, is_html)            
            if result:
                result_tables = result[0]
                result_url = result[1]

        except Exception as e:
            flash(e, "Error in SearchForm.search_text.data")

    return render_template("index.html", 
                            form=form,
                            title="Woodland Plantfinder Dev", 
                            template_folder='templates',
                            table=result_tables,
                            table_url = result_url,
                            search_term=search_term,
                            user_org = user_org,
                            initStatusElement=initStatusElement
                            )
