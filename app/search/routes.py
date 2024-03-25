from flask import Blueprint, jsonify, request, current_app, redirect, flash, url_for

from app.search import bp
from app.search.search import searchData

@bp.route('/', methods=["POST"], strict_slashes=False)
def search_data():

    # try deferred import 
    # from app.search import searchData

    search_term = request.form.get('search_text')

    # Call the searchData function
    is_html = True  # use to format detailed status in init_details db field
    result = searchData(search_term, is_html) if len(search_term) > 0 else None      

    if result:
        table = result[0] if len(result[0]) > 0 else None
        table_url = result[1] if len(result[1]) > 0 else None
    else:
        table = None 
        table_url = None 

    return jsonify({'table': table,
                    'table_url' : table_url}
                    )
