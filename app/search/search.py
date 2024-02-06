# Functions to match text with previously established text dictionary 
# Uses exact match with AND logic to match multiple terms
# requires previously initialized structure called dfDIct retrieved from app session

from flask import Flask, render_template,redirect,flash,url_for,session,Blueprint,current_app,jsonify, request
from flask_bcrypt import Bcrypt,generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import UserMixin,login_user,LoginManager,current_user,logout_user,login_required
from sqlalchemy.exc import IntegrityError,DataError,DatabaseError,InterfaceError,InvalidRequestError
from werkzeug.routing import BuildError
import logging
import base64
import io
import json
import time 
import pandas as pd
from datetime import datetime

# import gunicorn 
# from flask_session import Session
# from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.models import UserData, Tables, TableData
from app.main.forms import search_form

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
    2023-10-07 changed the AND sepoarator from "+" to " and "
    '''

    if isHTML:
        NL = '<br>'
    else:
        NL = '\r\n'

    # retrieve the user data
    myUserData = UserData.query.get(current_user.id)

    # retrieve a list of tables 
    my_tables = Tables.get_all_sorted()
    # init output string 
    theOutput = ""

    for tbl in my_tables:
        # retrieve the df
        tbl_df = TableData.get_dataframe(tbl.table_name)
        # retrieve the search cols 
        tbl_search_cols = TableData.get_search_columns(tbl.table_name)
        # get the first 5 rows to test 
        tbl_search_test = get_rows_as_list_of_lists(tbl_df, tbl_search_cols)
        # log results 
        for t in tbl_search_test:
            logging.info(t)
            theOutput = theOutput + str(t) + NL 

    # skip the old stuff
    if searchTerm and False:
        searchStrings = [x.strip() for x in searchTerm.split(' and ')]
        
        # get a list of line numbers with a text string match and print them 
        foundText = {}
        tmpFind = {}
        theOutput = ""

        # sorting and filering - group by vendor, show latest date 
        tmpDictSort = []
        for tFile,fString in dfDict.items():
            # get a standard date format
            if type(fString[1]) is list:
                myDate = datetime.strptime(fString[1][0], "%B %d, %Y")
            elif 'datetime' in str(type(fString[1])):
                myDate = fString[1]
            else:
                myDate = 'bad date format'
            
            # get the vendor 
            tmpDictSort.append([tFile.split('/')[2].upper(), myDate, tFile])

        tmpDictSort.reverse()
        # now select just the latest version of each supplier
        tmpDictDate = {x[0]:(x[1], x[2]) for x in tmpDictSort}
        tmpDictSupplier = list(tmpDictDate.keys())
        tmpDictSupplier.sort()
        
        for mySupplier in tmpDictSupplier:
            # get the date and supplier 
            myDate = tmpDictDate[mySupplier][0].strftime( "%B %d, %Y")
            myFile = tmpDictDate[mySupplier][1].split('/')[3]
            tFile = tmpDictDate[mySupplier][1]
            print(f"searching {mySupplier}: {tFile.split('/')[-1]}")

            # logic for AND search 
            for myTerm in searchStrings:
                tmpFind[myTerm] = findString(myTerm,dfDict[tFile][0])

            # now we have search results for each term - check for overlaps
            intersectFind = set.intersection(*[set(x) for x in tmpFind.values()]) 
            foundText[tFile] = list(intersectFind)

            if foundText.get(tFile): 
                # print(f"\nFound {searchTerm} in {tFile} modified {fString[1]}:\n")
                # [print("    ",x) for n,x in enumerate(dfDict[tFile][0].split('\n')) if n in foundText.get(tFile) ]
                theOutput += f"{NL}{NL}{mySupplier}: {myFile}{NL} modified {myDate} "
                tmpOutput =  ["    " + x for n,x in enumerate(dfDict[tFile][0].split('\n')) if n in foundText.get(tFile) ]            
                for myRow in tmpOutput:
                    theOutput += f"{NL}{myRow}"
            jnk = 0 

    return(theOutput)
