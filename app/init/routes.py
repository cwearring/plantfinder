import uuid
import threading
from datetime import datetime 

from flask import request, Response
from flask import current_app, jsonify, render_template, redirect, url_for
from flask_login import current_user

import os
from dotenv import load_dotenv
# for drop box data access - get env variables from .env file 
load_dotenv()
APP_KEY = os.environ.get('DROPBOX_APP_KEY')
APP_SECRET = os.environ.get('DROPBOX_APP_SECRET')
REFRESH_TOKEN = os.environ.get('DROPBOX_REFRESH_TOKEN')

import logging
# Configure logging
# logging.basicConfig(level=logging.INFO)

from app import db
from app.init import bp 
from app.models import ThreadComplete, Organization

from .extract_table import background_task, message_queue, Empty

@bp.route("/init", methods=["GET", "POST"], strict_slashes=False)
def index():
    # template_folder_value = get_absolute_template_folder(bp)
    # x = bp.template_folder   y = bp.root_path
     
    # for sync testing 
    # results = save_all_file_tables_in_dir('../' + dirpath)

    app = current_app._get_current_object()
    
    # get user and org data 
    userid = current_user.id
    user_org = Organization.query.get(current_user.org_id)
    useDropbox = user_org.is_dropbox 
    dirpath = user_org.dirpath 

    # for streaming over message queue
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

