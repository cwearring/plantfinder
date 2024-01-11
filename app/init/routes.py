
from flask import Flask, Blueprint, current_app, jsonify, session, render_template,redirect,flash,url_for
import uuid

from app.init import bp 
from app.models import User, db, SessionData, UserData

# refresh inventory is called from button event via javascript 
@bp.route('/init/', methods=('GET','POST'))
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

