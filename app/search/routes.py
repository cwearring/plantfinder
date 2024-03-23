from flask import Blueprint, jsonify, request, current_app, redirect, flash, url_for

from app.search import bp
from app.search.search import searchData

@bp.route('/search', methods=["POST"], strict_slashes=False)
def search_data():

    # try deferred import 
    # from app.search import searchData

    data = request.json
    search_term = data.get('searchQuery')

    # Call the searchData function
    is_html = True  # Set this according to your requirements
    result = searchData(search_term, is_html)

    if result:
        table = result[0]
        table_url = result[1]

    return jsonify({'table': table,
                    'table_url' : table_url}
                    )
