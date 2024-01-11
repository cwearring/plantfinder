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
import base64
import io
import json
import time 
# import datetime
from datetime import datetime

# import gunicorn 
# from flask_session import Session
# from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.models import User, db, SessionData, UserData
from app.main.forms import search_form

def findString(token:str, myString:str):
    tst = [n for n,x in enumerate( myString.split('\n')) if token.lower() in x.lower()]
    return(tst)

def searchData(searchTerm:str, isHTML = True ):
    '''
    2023-10-07 changed the AND sepoarator from "+" to " and "
    '''

    if isHTML:
        NL = '<br>'
    else:
        NL = '\r\n'

    # retrieve the dataframe saved from initSearchDropBox()
    myUserData = UserData.query.get(current_user.id)
    dfDict = myUserData.data.get('dfDict')
    
    # decompose the search term and return the found text  
    if searchTerm and dfDict:
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
    else:
        theOutput = "No search term provided or dfDict not initialized"

    return(theOutput)

