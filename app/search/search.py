# Functions to match text with previously established text dictionary 
# 2024-02-07 Uses exact match with AND logic to match multiple terms

from flask import Flask, render_template,redirect,flash,url_for,session,Blueprint,current_app,jsonify, request
from sqlalchemy.exc import SQLAlchemyError, IntegrityError,DataError,DatabaseError,InterfaceError,InvalidRequestError
from werkzeug.routing import BuildError
import logging
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
    # 2024-02-08 Have to add smarts to differentiate shrubs, perennials, general(annuals?)
    all_vendors = Tables.get_unique_vendors()
    most_recent_by_vendor = [Tables.get_most_recent_by_vendor(v) for v in all_vendors]

    # init output as a dict of dicts 
    theOutput = {}
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

            # Create a mask where each search term is checked individually, and only rows where all terms are found are marked True
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
                output_df = filtered_df[included_columns]
                # Remove any columns for which all row values have string = 'empty' or are empty
                # Find columns where all values are 'empty'
                columns_to_drop = [col for col in output_df.columns if (output_df[col] == 'empty').all()]

                # Drop these columns from the DataFrame
                output_df.drop(columns=columns_to_drop, inplace=True)

                # Transform the DataFrame into a dict of lists directly
                dict_of_lists = output_df.to_dict(orient='list')

                # populate the return structure with the filtered columns
                theOutput[f"{tbl.vendor}:   {tbl.table_name}"]=dict_of_lists

    except Exception as e:
        logging.error(f"General Exception: {e}")
        return "Error: An unexpected error occurred."

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
