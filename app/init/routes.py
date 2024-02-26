import uuid
import threading
from datetime import datetime 

from flask import Blueprint, request, Response
from flask import session, current_app, jsonify, render_template, redirect, flash, url_for
from flask_login import current_user

# for drop box data access - get env variables from .env file 
import os
from dotenv import load_dotenv
load_dotenv()
APP_KEY = os.environ.get('DROPBOX_APP_KEY')
APP_SECRET = os.environ.get('DROPBOX_APP_SECRET')
REFRESH_TOKEN = os.environ.get('DROPBOX_REFRESH_TOKEN')

# for more secure file handling
from werkzeug.utils import secure_filename

# for logging 
import logging
# Configure logging
# logging.basicConfig(level=logging.INFO)

from app import db
from app.init import bp 
from app.models import User,  SessionData, UserData, ThreadComplete

from .extract_table import background_task, message_queue, Empty

# refresh inventory is called from button event via javascript 
@bp.route('/init/', methods=['GET','POST'])
def init_data():

    jnk = 0

    # delayed import 
    from app.init.init import initSearchDropBox

    #manually restrict scope for testing
    subDirFilter = ['Canadale', 'AVK']

    # Check if session contains a unique identifier
    if 'session_id' not in session:
    # Generate a unique identifier and store it in the session
        session['session_id'] = str(uuid.uuid4())
    
    # retrieve the local variable 
    session_id = session['session_id']

    # initialize the search data using an archived DropBox refresh token
    init_status = initSearchDropBox(onlySubdir=subDirFilter)

    # Store the session data in the database
    session_data = SessionData(id=session_id, data={'init_status': init_status["status"],
                                                    'filesfound': init_status['filesfound'],
                                                    'filesloaded': init_status['filesloaded']})
    db.session.merge(session_data)
    db.session.commit()

    # temp = jsonify(init_status=init_status)
    # temp2 = init_status['status']
    return jsonify(status=init_status["status"], 
                   filesfound=init_status['filesfound'], 
                   filesloaded=init_status['filesloaded'])

@bp.route("/init2/", methods=["GET", "POST"], strict_slashes=False)
def index():
    # template_folder_value = get_absolute_template_folder(bp)
    # x = bp.template_folder   y = bp.root_path
     
    # dirpath = request.form['dirpath']
    dirpath = './OrderForms' # 2024-01-29 hardcode for testing 
    
    # for sync testing 
    # results = save_all_file_tables_in_dir('../' + dirpath)

    # for streaming over message queue
    app = current_app._get_current_object()
    userid = current_user.id
    useDropbox = True 

    threading.Thread(target=background_task, args=(app, dirpath, userid, useDropbox )).start()

    return jsonify(status="started")

@bp.route('/status')
def status():
    if current_user.is_authenticated:
        return render_template('status.html', current_user=current_user)
    else:
        return redirect(url_for('auth.login'))

@bp.route('/check_task',methods=["POST"], strict_slashes=False)
def check_task():
    data = request.get_json()
    userid = data.get('user_id', None)  # Use provided user_id from request

    task_complete = ThreadComplete.is_task_complete(id=userid)

    # print(f"Task Complete = {task_complete} ")
    return jsonify({'task_complete': task_complete})

@bp.route('/status_stream',methods=["GET", "POST"])
def status_stream():
    def generate():
        while True:
            try:
                message = message_queue.get(timeout=10)  # Adjust timeout as needed
            except Empty:
                yield "data: ping\n\n"  # Send a ping or keep-alive message
                continue
            if message is None:
                break
            yield f"data: {message}\n\n"            

    return Response(generate(), mimetype='text/event-stream')

