# Functions to match text with previously established text dictionary 
# 2024-02-07 Uses exact match with AND logic to match multiple terms

from flask import Flask, render_template,redirect,flash,url_for,session,Blueprint,current_app,jsonify, request
from sqlalchemy.exc import SQLAlchemyError, IntegrityError,DataError,DatabaseError,InterfaceError,InvalidRequestError
from werkzeug.routing import BuildError
import logging
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# import gunicorn 
# from flask_session import Session
# from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.models import Organization, Tables, TableData

# get settings from the .env file 
load_dotenv()
# create a cutoff_date to use for the table reads 
cutoff_date_string = os.getenv('CUTOFF_DATE', None)
cutoff_date = datetime.combine(datetime.strptime(cutoff_date_string, '%Y-%m-%d').date(), datetime.min.time())

def findString(token:str, myString:str):
    tst = [n for n,x in enumerate( myString.split('\n')) if token.lower() in x.lower()]
    return(tst)

def get_rows_as_list_of_lists(df, column_names):
    # Ensure only the specified columns are selected, if they exist in the DataFrame
    if not all(item in df.columns for item in column_names):
        raise ValueError("One or more specified columns do not exist in the DataFrame")
    
    # Select specified columns and the first 5 rows
    selected_data = df[column_names].head(5)
    
    # Convert to list of lists
    rows_list = selected_data.values.tolist()
    
    return rows_list

def model_to_dataframe(model):
    """
    Convert a Flask-SQLAlchemy model into a pandas DataFrame.

    Parameters:
    - model: The Flask-SQLAlchemy model class to convert.

    Returns:
    - A pandas DataFrame containing the data from the model's table.
    """
    # Query all data from the model's table
    query_result = model.query.all()

    # Convert the query result to a list of dictionaries
    # Each dictionary corresponds to a row in the table
    data = [row.__dict__ for row in query_result]

    # Remove '_sa_instance_state' from dictionaries, which is added by SQLAlchemy
    for row in data:
        row.pop('_sa_instance_state', None)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    return df

def searchData(searchTerm:str, isHTML = True ):
    '''
    2023-10-07 changed the AND separator from "+" to " and "
    '''

    # Define a helper function - Adjusted approach to avoid the error:
    def check_all_terms(row, search_terms):
        return all(row.str.contains(term, case=False, na=False).any() for term in search_terms)
    
    if isHTML:
        NL = '<br>'
    else:
        NL = '\r\n'

    # retrieve the user data
    searchStrings = [x.strip() for x in searchTerm.split(' and ')]
    # searchStrings = [x.strip() for x in searchTerm.split(' or ')]

    # retrieve a list of tables 
    all_tables = Tables.get_all_sorted()
    
    # delete the rows with the same vendor to retain just one with latest file_modifed 
    # 2024-03-21 added filter for perennials - get_most_recent_by_vendor returns tuple ()
    all_vendors = Tables.get_unique_vendors()
    # get latest with and without perennial in filename 
    # most_recent_by_vendor = [Tables.get_most_recent_by_vendor(v, str_token='erennia') for v in all_vendors]

    # temp hack on 2024-06-05 - get all the files since there are multiple valid files per vendor 
    most_recent_by_vendor = [Tables.get_all_by_vendor(v, cutoff_date) for v in all_vendors]

    # unpack the tuples to a single list removing empty entries 
    most_recent_by_vendor = [x for t in most_recent_by_vendor for x in t if x]

    # init output as a dict of dicts 
    theOutput = {}
    theUrls = {} # 2024-03-21 - added to allow dropbox URL link in search results 

    # top key = name of table 
    # sub keys are headings -> list of values for that heading 

    try:
        for tbl in most_recent_by_vendor:
            logging.info(f"{tbl.table_name}")

            # retrieve the df
            tbl_df = TableData.get_dataframe(tbl.table_name)
            tbl_df.fillna('Unknown', inplace=True)

            # retrieve the search cols 
            tbl_search_cols = TableData.get_search_columns(tbl.table_name)

            # Filter rows where any of the specified columns contain any of the 'search_terms'
            mask_any = tbl_df[tbl_search_cols].astype(str).apply(
                lambda x: x.str.contains('|'.join(searchStrings), case=False, na=False)).any(axis=1)
            filtered_df = tbl_df[mask_any]

            # Create a mask where each search term is checked individually, 
            # and only rows where all terms are found are marked True
            # Applying the helper function across the DataFrame:
            mask_all = tbl_df[tbl_search_cols].astype(str).apply(
                lambda x: check_all_terms(x, searchStrings),
                axis=1
            )
            filtered_df = tbl_df[mask_all]

            if len(filtered_df) > 0:
                # Select columns not in the excluded list
                included_columns = [col for col in filtered_df.columns if col.lower() 
                                    not in ['tablename','text','order','total']]

                # Create a new DataFrame excluding the specified columns
                output_df = filtered_df[included_columns].copy()
                
                # Remove any columns for which all row values have string = 'empty' or are empty
                # Find columns where all values are 'empty'
                columns_to_drop = [col for col in output_df.columns if (output_df[col] == 'empty').all()]

                # Drop these columns from the DataFrame
                output_df.drop(columns=columns_to_drop, inplace=True)

                # Transform the DataFrame into a dict of lists directly
                dict_of_lists = output_df.to_dict(orient='list')

                # populate the return structure with the filtered columns
                theOutput[f"{tbl.vendor}:   {tbl.table_name}"]=dict_of_lists
                # a matching structure with the dropbox URLs if they exist 
                theUrls[f"{tbl.vendor}:   {tbl.table_name}"]= tbl.file_dropbox_url

    except Exception as e:
        logging.error(f"General Exception: {e}")
        return "Error: An unexpected error occurred."

    return (theOutput, theUrls)
