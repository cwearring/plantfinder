
from flask import Blueprint, jsonify, session, render_template, request, current_app, redirect, flash, url_for

from app.search import bp
from app.main.forms import SearchForm
from app.models import User, db

@bp.route('/search_data/', methods=("GET", "POST"), strict_slashes=False)
def search_data():

    # try deferred import 
    from app.search import searchData

    # create an instance from the form class
    form = SearchForm()

    if form.validate_on_submit():
        try:
            search_term = form.search_text.data
            # Call the searchData function
            is_html = True  # Set this according to your requirements
            result = searchData(search_term, is_html)
            return result
        except Exception as e:
            flash(e, "Error in SearchForm.search_text.data")

    return render_template("home.html",
        form=form,
        text="Search",
        title="Search",
        btn_action="Search String"
        )

@bp.route('/xxxxsearch_data',  methods=("GET", "POST"))
def xxxsearch_data():
    # deferred load 
    from app.search import searchData

    search_text = request.form.get('search_text')
    results = searchData(search_text)
    return jsonify(results=results)

