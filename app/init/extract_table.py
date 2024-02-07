'''
https://pymupdf.readthedocs.io/en/latest/page.html#Page.find_tables
https://pypi.org/project/PyMuPDF/1.23.9/
PYDEVD_WARN_EVALUATION_TIMEOUT environment variable to a bigger value

https://www.docugami.com/pricing

(clip=None, vertical_strategy='lines', horizontal_strategy='lines', 
vertical_lines=None, horizontal_lines=None, 
snap_tolerance=3, snap_x_tolerance=None, snap_y_tolerance=None, 
join_tolerance=3, join_x_tolerance=None, join_y_tolerance=None, 
edge_min_length=3, 
min_words_vertical=3, min_words_horizontal=1, 
intersection_tolerance=3, intersection_x_tolerance=None, intersection_y_tolerance=None, 
text_tolerance=3, text_x_tolerance=3, text_y_tolerance=3)

row_rgb = {f'{n:02d}':(# page.get_pixmap(clip = row.cells[0]).colorspace.this.Fixed_RGB,
                        # page.get_pixmap(clip = row.cells[0]).color_topusage()[0],
                        
                        sum(1 for item in tbl.extract()[n] if len(item) > 0),
                        tbl.extract()[n][0])
            for n,row in enumerate (tbl.rows)}

# freq = [len(row.cells) for row in tbl.rows]
# counts = {item:freq.count(item) for item in freq}
# pp.pprint(counts)

# for n,row in enumerate (tbl.rows):
#    rgb = page.get_pixmap(clip = row.cells[0]).px.colorspace.this.Fixed_BGR

https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_Structured_RAG.ipynb?ref=blog.langchain.dev 
https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector
https://levelup.gitconnected.com/a-guide-to-processing-tables-in-rag-pipelines-with-llamaindex-and-unstructuredio-3500c8f917a7


https://pymupdf.readthedocs.io/en/latest/recipes-text.html#how-to-extract-text-in-natural-reading-order

https://pymupdf.readthedocs.io/en/latest/recipes-text.html#how-to-analyze-font-characteristics


'''        

# take directory path as input and create entries in sqlalchemy db 
import os
import ast
import re
import uuid
import hashlib
from dotenv import load_dotenv

# fitz is the PDF parser with table recognize capabilities 
import fitz
import pandas as pd
from datetime import datetime 
from difflib import SequenceMatcher
from typing import List, Optional

# for drop box data access
import base64
import requests
import dropbox

# defined in .env file 
APP_KEY = os.environ.get('DROPBOX_APP_KEY')
APP_SECRET = os.environ.get('DROPBOX_APP_SECRET')
REFRESH_TOKEN = os.environ.get('DROPBOX_REFRESH_TOKEN')

# for more secure file handling
from werkzeug.utils import secure_filename

# for logging 
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# LlmaIndex manages data and embeddings 
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor

# lower level functions for creating nodes 
from llama_index.schema import TextNode

# modules to create an http server with API
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
# https://docs.sqlalchemy.org/en/20/core/engines.html
from sqlalchemy import  Column, inspect
from sqlalchemy.exc import SQLAlchemyError


# queues for streaming updates during init 
from queue import Queue, Empty
import threading
# Global message queue
message_queue = Queue()

# get functions and variables created by __init__.py
from app import db, create_app
from app.models import ThreadComplete, UserData, Tables, TableData

# holding area for unused 
def best_header_word_match(header_word):
    # define a list of header words from the docs 
    header_words = ['Product', 'Variety', 'Size', 'Colour', 'Order Qty', 'Cost', 'Description', 'Code', 'Name',\
                'Category','Your Price', 'Price', 'Status', 'Zone', 'Catalog $', 'Pots/Tray', 'Amount',\
                'WH', 'Botanical Name', 'E-LINE', 'Available','Order', 'Total', 'PIN', 'UPC','Latin Name',\
                'Available Units','QTY', 'Notes','Avail QTY','Order Qty','Plant Name','Common Name','Sale Price',\
                'Pot Size','List','Net','Comments','AVL','Sku','Case Qty','Packaging', "Pots Ordered", 'SIZE 1', 'SIZE 2']
 # Initial check for null string
    if not header_word:
        return None
    
    # find the best match from header_words 
    tmp = {n:SequenceMatcher(a=header_word.lower(), b=hd.lower()).ratio() for n,hd in enumerate(header_words)}
    header_word_key = max(tmp, key=tmp.get)
    match_value = tmp[header_word_key]
    header_word_key = max(tmp, key=tmp.get)
    match_header_word = header_words[header_word_key]

    return match_header_word

def generate_hash(text):
    hash_object = hashlib.sha256(text.encode())
    return hash_object.hexdigest()

def string_to_list(string: str) -> List:
    """
    Converts a string representation of a list into an actual Python list object.

    Parameters:
    - string (str): A string that represents a list, e.g., "[1, 2, 3]".

    Returns:
    - List: The list object represented by the input string.

    Raises:
    - ValueError: If the input string does not represent a list or if there's an error
      in converting the string to a list.
    """
    
    try:
        result = ast.literal_eval(string)
    except (SyntaxError, ValueError) as e:
        # Catching specific exceptions related to literal_eval failures
        raise ValueError(f"Error converting string to list: {e}")
    
    if not isinstance(result, list):
        raise ValueError("The evaluated expression is not a list")

    return result

def extract_text_within_brackets(input_string: str) -> List[str]:
    """
    Extracts and returns all text found within square brackets in a given string.

    Parameters:
    - input_string (str): The string from which to extract text within square brackets.

    Returns:
    - List[str]: A list of strings found within square brackets. If no text is found
      within brackets, returns an empty list.

    Examples:
    >>> extract_text_within_brackets("Example [text] within [brackets].")
    ['text', 'brackets']
    >>> extract_text_within_brackets("No brackets here.")
    []
    """
    
    if not isinstance(input_string, str):
        raise ValueError("Input must be a string.")
    
    # Define the regex pattern to find text within square brackets
    pattern = r'\[(.*?)\]'

    # Use re.findall() to find all occurrences in the string
    matches = re.findall(pattern, input_string)

    return matches

def create_class_from_df(df, class_name, p_key):
    '''
    Dynamically create a SQLAlchemy class from a dataframe 
    '''
    try:
        type_mapping = {
            'int64': db.Integer,
            'float64': db.Float,
            'object': db.String  # Assuming all 'object' types in this DataFrame are strings
        }

        # attributes = {col: Column(type_mapping[str(df[col].dtype)]) for col in df.columns}

        # Adding a primary key column
        # attributes = {'id': Column(db.Integer, primary_key=True, autoincrement=True)}
        attributes = {
            '__tablename__': class_name.lower(),  # Table name, typically lowercase
            p_key : Column(db.String(64), primary_key=True),
            '__table_args__': {'extend_existing': True}  # Add this line
        }

        # Add columns from DataFrame
        for col in [c for c in df.columns if c != p_key]:
            attributes[col] = Column(type_mapping[str(df[col].dtype)])

        return type(class_name, (db.Model,), attributes)

    except Exception as e:
        # Optionally, log the error here - extend_existing=True
        # log.error(f"Error in save_class_in_session: {str(e)}")

        # Raise an exception with a descriptive message
        raise ValueError(f"An error occurred in create_class_from_df: {str(e)}")
    
def save_class_in_session(df, class_name, p_key):
    '''
        make a sqlalchemy ORM class for each dataframe - add column = id:int as primary key
    '''
    jnk=0 #     print(current_app.name)

    """# save a sqlalchemy ORM class for each dataframe - add column = id:int as primary key
    status = save_class_in_session(df=file_table, class_name=filetoken, p_key=p_key)

    if status:
        logging.info(f"Created ORM class and db table {filetoken}")
        # yield f"Created ORM class and db table {filetoken} at {datetime.now():%b %d %I:%M %p}"
        yield f"Updated {len(file_table)} rows for {file_data['filename']} at {datetime.now():%b %d %I:%M %p}"
    else:
        logging.info(f"Hit an error save_class_in_session for {file_data['filename']}")
        yield f"Error in save_class_in_session() for {file_data['filename']}"
    """
    try:
        #with current_app.app_context():
        # Get an inspector object from the database connection
        inspector = inspect(db.engine)

        # Create ORM class from DataFrame
        DynamicClass = \
            create_class_from_df(df, class_name, p_key )

        # Check if the table already exists to avoid overwriting
        if not inspector.has_table(DynamicClass.__tablename__):
                DynamicClass.__table__.create(bind=db.engine)

        # Iterate over DataFrame rows
        for _, row in df.iterrows():
            # Create an instance of the ORM class
            obj = DynamicClass(**row.to_dict())

            # Add the instance to the session
            db.session.merge(obj)
            
        # Commit the session to save changes to the database
        db.session.commit()

        return True
    
    except Exception as e:
        # Optionally, log the error here
        # log.error(f"Error in save_class_in_session: {str(e)}")

        # Return a status indicating an error occurred and include error details
        raise ValueError(f"An error occurred in save_class_in_session: {str(e)}")
    
        return {"status": "error", "message": str(e)}
    
    return 

# start of useful functions 
def allowed_file(filename):
    '''
    CHeck filename against allowed extensions 
    '''
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'xls', 'xlsx'}

    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def most_frequent_integer(int_list: List[int]) -> Optional[int]:
    """
    Finds the most frequently occurring integer in a list.
    
    Parameters:
    - int_list (List[int]): A list of integers to analyze.
    
    Returns:
    - Optional[int]: The integer that appears most frequently in the list. If there are multiple integers
      with the same highest frequency, one of them will be returned. Returns None if the list is empty
      or contains no integers.
    
    Raises:
    - ValueError: If the input list contains non-integer elements.
    - ValueError: If the input is not a list.
    """
    
    if not isinstance(int_list, list):
        raise ValueError("Input must be a list.")
    
    if len(int_list) == 0:
        return None  # Or raise an exception, depending on desired behavior
    
    if not all(isinstance(x, int) for x in int_list):
        raise ValueError("The list must contain only integers.")

    frequency = {}
    for num in int_list:
        frequency[num] = frequency.get(num, 0) + 1

    most_frequent = max(frequency, key=frequency.get, default=None)

    return most_frequent

def is_file_dropbox(dropboxMeta):
    '''
    Check a drop box object id and test if it is a file
    Returns True if object is file, False otherwise
    '''
    return isinstance(dropboxMeta,dropbox.files.FileMetadata)

def parse_fullfilename(full_filename:str = None):
    '''
    Parse a full file URI into components
    : returns a dictionary 
        'fullfilename': full_filename,
        'dirpath':dirpath, 
        'vendor':vendor,
        'filename':filename, 
        'filetoken': filetoken, # secure_filename(filename.split('.')[0])
        'filetype':filetype
    '''
    try:
        # get the filename and directory path and file type
        dirpath = '/'.join(full_filename.split('/')[0:-1])    
        vendor = dirpath.split('/')[-1]
        filetype = secure_filename(full_filename).split('.')[-1]
        filename = full_filename.split('/')[-1]
        filetoken = secure_filename(filename.split('.')[0])

        return {'fullfilename': full_filename,'dirpath':dirpath,'filename':filename, 
                'filetoken':filetoken, 'filetype':filetype, 'vendor':vendor }
    except:
        return {'fullfilename':None,'dirpath':None, 'filename':None, 'filetoken':None, 'filetype':None, 'vendor':None  }

def get_dropbox_accesstoken_from_refreshtoken(REFRESH_TOKEN, APP_KEY, APP_SECRET):
    '''
    ### Get the auth token from a valid refresh token, Key and Secret
    returns an authorization token for dropbox, error otherwise 

    https://www.dropboxforum.com/t5/Dropbox-API-Support-Feedback/Get-refresh-token-from-access-token/m-p/596755/highlight/false#M27728
    https://www.dropbox.com/developers/documentation/http/documentation#authorization 
    https://www.dropbox.com/oauth2/authorize?client_id=j27yo01f5f8w304&response_type=code&token_access_type=offline
    
    i have to get the access code dynamically from the refresh token
    https://developers.dropbox.com/oidc-guide 

    This gets a refresh token if we pass a valid AUTHORIZATIONCODEHERE
    We only need to do this once to get a persistent refresh token  
    curl https://api.dropbox.com/oauth2/token \
        -d code=AUTHORIZATIONCODEHERE \
        -d grant_type=authorization_code \
        -u APPKEYHERE:APPSECRETHERE​

    This gets the authcode from the refresh token
    curl https://api.dropbox.com/oauth2/token \
        -d refresh_token=REFRESHTOKENHERE \
        -d grant_type=refresh_token \
        -d client_id=APPKEYHERE \
        -d client_secret=APPSECRETHERE
    '''
    try:

        if not all([APP_KEY, APP_SECRET, REFRESH_TOKEN]):
            logging.error("Missing required Dropbox configuration.")
            return "Error: Missing configuration."

        BASIC_AUTH = base64.b64encode(f'{APP_KEY}:{APP_SECRET}'.encode()).decode()

        data = {
            "refresh_token": REFRESH_TOKEN,
            "grant_type": "refresh_token",
            "client_id": APP_KEY,
            "client_secret": APP_SECRET
        }

        response = requests.post('https://api.dropboxapi.com/oauth2/token', data=data)

        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            logging.error(f"Dropbox API Error: {response.status_code} - {response.text}")
            return "Error: Failed to retrieve access token."

    except requests.RequestException as e:
        logging.error(f"Request Exception: {e}")
        return "Error: Network error occurred."

    except Exception as e:
        logging.error(f"General Exception: {e}")
        return "Error: An unexpected error occurred."

def get_dropbox_filenames(dbx, startDir:str):
    """
    Walks down all subdirectories from a top-level directory and collects filenames.

    :param startDir: The top-level directory to start the traversal from.
    :return: A Dict of filenames found in all subdirectories.
    """
    filesFound = {} # key = filepath, value = dbx meta entity

    # Recursive helper function
    def walk_dir_recursive(dbx, current_dir:str):
        try:
            entries = dbx.files_list_folder(current_dir).entries
            for entry in entries:
                if is_file_dropbox(entry):
                    logging.info(f"\nIn {f'/{current_dir}'} found File: {entry.name}")
                    filesFound[entry.path_lower] = entry
                else: 
                    logging.info(f"\nRecursing into {f'/{entry.path_lower}'} ")
                    walk_dir_recursive(dbx, entry.path_lower)  # Recursion step

        except PermissionError:
            logging.error(f"Permission denied: {current_dir}")
            logging.error(f"Error accessing Dropbox: {e}")
        except FileNotFoundError:
            logging.error(f"File not found: {current_dir}")
            logging.error(f"Error accessing Dropbox: {e}")
        except OSError as e:
            logging.error(f"OS error: {e}")
            logging.error(f"Error accessing Dropbox: {e}")

        return(filesFound)
    
    walk_dir_recursive(dbx, startDir)

    return filesFound

def get_filenames_in_directory(top_dir):
    """
    Walks down all subdirectories from a top-level directory and collects filenames.

    :param top_dir: The top-level directory to start the traversal from.
    :return: A list of filenames found in all subdirectories.
    """
    filesFound = {}

    # Recursive helper function
    def walk_dir_recursive(current_dir):
        try:
            for entry in os.scandir(current_dir):
                if entry.is_file() and entry.name[0:2] != '~$':
                    filesFound[entry.path] = entry.path
                    #filenames.append(entry.path)
                elif entry.is_dir():
                    walk_dir_recursive(entry.path)  # Recursion step
        except PermissionError:
            logging.error(f"Permission denied: {current_dir}")
            logging.error(f"Error accessing Dropbox: {e}")
        except FileNotFoundError:
            logging.error(f"File not found: {current_dir}")
            logging.error(f"Error accessing Dropbox: {e}")
        except OSError as e:
            logging.error(f"OS error: {e}")
            logging.error(f"Error accessing Dropbox: {e}")

    walk_dir_recursive(top_dir)

    return filesFound

def get_firstpage_tables_as_list(doc=None):
    '''
    Get the first page of a fitz doc as a list of lists 
    : tbl_out is the list of lists
    : col_count is the number of columns (max) 
    :table_strategy(lines or text )is the fitz table parsing strategy used'''

    if not(doc):
        raise ValueError("fitz document not set - get_firstpage_tables_as_list")

    # focus on the first page 
    table_strategy = 'lines'
    tbls = doc[0].find_tables(vertical_strategy='lines', horizontal_strategy='lines')
    if len(tbls.tables) ==0: # did not find tables by grid, try spaces 
        tbls = doc[0].find_tables(vertical_strategy='text', horizontal_strategy='text')
        table_strategy = 'text'

    # merge the tables 
    tbl_out = []
    col_counts = []

    for tbl in tbls.tables:
        tbl_out.extend(tbl.extract())
        col_counts.append(tbl.col_count)

    col_count = most_frequent_integer(col_counts)

    return tbl_out, col_count, table_strategy, tbls

def get_bestguess_table_header(table_as_list = None):
    '''
    Uses embedding and LLM to guess the best header row
    : returns a dictionary
        'header_rownum' : header_rownum,
        'header_node_text' : header_node_text,
        'header_raw' : header_raw,
        'p_key' : p_key, a variable with the name of the sqlalchemy table primary key 
        'header_guess' : header_guess
        'plantname_in_header':plantname_in_header
    '''
    try:
        # get the best guess at the header row
        if len(table_as_list) > 0:

            # guess the most likely number of columns from the most frequent table column count 
            numcols = most_frequent_integer([len(x) for x in table_as_list])

            # find the first row with the most frequent number of columns- skip header stuff
            for n,row in enumerate(table_as_list):
                if len(row) == numcols:
                    table_row_start_guess = n
                    break

            # create text nodes with one node per row in tmp 
            table_nodes = [TextNode(text=f"'{t}'", id_=n) for n,t in enumerate(table_as_list)]
            table_index = VectorStoreIndex(table_nodes)

            # create the query engine with custom config to return just one node
            # https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/root.html#query-engine 
            query_engine = table_index.as_query_engine(
                similarity_top_k=1,
                vector_store_query_mode="default",
                alpha=None,
                doc_ids=None,
            )
            # request gpt's best guess at the row to use for table headings 
            oneshotprompt=f"""Return {numcols}  table column headings from this 
            price sheet of plants as a python list of length {numcols}.

            Return an existing row. Do not make up rows.
            """

            multishotprompt=f"""Return {numcols} table column headings for this price sheet 
            of plants with columns as a python list of length {numcols}. 

            Example table column headings:
            5 column  ['Name', 'Latin Name', 'Price', 'Available Qty', 'Order Qty']
            8 column  [ "Product","SIZE1","SIZE2","PRICE", "AVL",  "COMMENTS", "ORDER", "Total"]
            7 column  ["Category", "WH", "Code", "Botantical Name", "size", "Price", "Available"]

            Return an existing row. Do not make up rows.
            """
            
            response = query_engine.query(multishotprompt)

            # get info from the header row
            header_rownum = int(response.source_nodes[0].id_)
            header_node_text = response.source_nodes[0].text
            header_raw = table_as_list[header_rownum]
            
            # correct for special case failure of table header guess 
            # Total Hack - tried using the GPT but it failed miserably
            if len(header_raw) == numcols: 
                # we have the right number of columns
                if header_rownum-table_row_start_guess < 10:
                    # we did not pick a row deep in the table
                    header_guess = ['TableName'] + [str(x).replace(' ','').replace('\n','_').replace('(','_').replace(')','')
                                            for x in header_raw] + ['Text']
                else:
                    # we think the row is too far down in the table 
                    header_guess = ['TableName'] + [f'col_{n}' for n in range(0,numcols)] + ['Text']
                    header_rownum = table_row_start_guess # start of the frequently ocurring table width 
            else:
                # this excludes header guesses with a different number of headings 
                header_guess = ['TableName'] + [f'col_{n}' for n in range(0,numcols)] + ['Text']
                header_rownum = table_row_start_guess # start of the frequently ocurring table width 

            best_guess={    
            'header_rownum' : header_rownum,
            'header_node_text' : table_nodes[header_rownum].text,
            'header_raw' : header_raw,
            'header_guess' : header_guess,
            'plantname_in_header': 'NotUsed'
            }
            
            return best_guess
        else:
            return None
    except:
        return None
        pass

def get_bestguess_table_search_columns(file_table):
    # Use the table to guess which columns have plant names and descriptions
    # submit with first few rows to guess columns having plant names or botantical names 
    try:
        # Generate the desired dictionary with keys as column names
        col_dict = {
            column_name: ', '.join(file_table[column_name].head(5).astype(str)) 
            for column_name in file_table.columns if column_name not in
            ['id_hash', 'TableName']
        }

        # create text nodes with one node per dict entry
        dict_nodes = [TextNode(text=f"Dict Key: {str(k)} has values: {str(v)}", id_=k) 
                        for k,v in col_dict.items()]
        dict_index = VectorStoreIndex(dict_nodes)

        # create the query engine 
        query_engine = dict_index.as_query_engine(
            similarity_top_k=3,
            vector_store_query_mode="default", alpha=None, doc_ids=None )

        # get the gpt's best guess at the columns containing plant names and descriptions
        oneshotprompt=f"""You are an expert on garden center plants and their botantical names.
        Return the dictionary keys that contain descriptions of plants or botanical names. 
        Return only existing keys. Do not explain. Just return the key.
        """

        response = query_engine.query(oneshotprompt)
        # df_search_cols = response.response.split(',')
        df_search_cols = list(response.metadata.keys())
        jnk = 0

        return df_search_cols

    except ValueError as e:
        print(f"Error: {e}")
        return None
    
def get_file_table_pdf(file_data:dict = None):
    '''
    Input: Directory path and filename 
    Output: Dict
        dataframe 
        table_header = {'filename': filename,
                    'numcols' : numcols,
                    'header_rownum':header_rownum,
                    'header_guess':header_guess, 
                    'header_raw':header_raw, 
                    'header_node_text':header_node_text}
        name of table's primary key 
    '''
    try:
        # check if the file exists 
        full_filename = file_data.get('fullfilename')

        # get the best guess at the header row
        if len(full_filename) > 0:
            # PDF: read the doc and get the first page tables
            doc = fitz.open(full_filename)
            tmp, numcols, tbl_strategy, tbls = get_firstpage_tables_as_list(doc)

            # get a dict using GPT to guess the header row  
            best_guess = get_bestguess_table_header(tmp)

            # merge header with the table from the first page 
            table_data = [ [file_data.get('filename')] +  row + [f"{row}"] for row in tmp[best_guess.get('header_rownum')+1:] ]
            
            # add the tables from the rest of the pages using the same tbl_strategy 
            # for pagenum in range(1,doc.page_count):
            for pagenum in range(1,min(200,doc.page_count)):
                logging.info(f"Processing page {pagenum} of {doc.page_count}")

                tbls = doc[pagenum].find_tables(vertical_strategy=tbl_strategy, horizontal_strategy=tbl_strategy)
                # exclude rows identical to the header row 
                tbl_page = [row for tbl in tbls.tables for row in tbl.extract() 
                            if row != best_guess.get('header_raw')]
                # add rows with p_key of entire row as text and add filename and text string 
                table_data.extend([ [file_data.get('filename')] + row + [f"{row}"] for row in tbl_page])
                jnk = 0 # for debugging

            # create a pandas dataframe 
            df = pd.DataFrame(table_data, columns = best_guess.get('header_guess'))
            # df.set_index('Table', inplace=True, drop=False)

            # save the file header info
            table_header = {'filename': file_data.get('filename'),
                            'numcols' : len(df.columns),
                            'header_rownum':best_guess.get('header_rownum'),
                            'header_guess':best_guess.get('header_guess'), 
                            'header_raw':best_guess.get('header_raw'), 
                            'header_node_text':best_guess.get('header_node_text'),
                            'plantname_in_header':best_guess.get('plantname_in_header')}
                        
            return df, table_header
        else:
            logging.error('No table found with fitz parse by grid or text ')
            return None, None
    except:
            pass

def get_file_table_xls(file_data = None):
    '''
    Input: Directory path and filename 
    Output: Dict
        dataframe 
        table_header = {'filename': filename,
                    'numcols' : numcols,
                    'header_rownum':header_rownum,
                    'header_guess':header_guess, 
                    'header_raw':header_raw, 
                    'header_node_text':header_node_text}
        name of table's primary key 
    '''
    try:
        # check if the file exists 
        full_filename = file_data.get('fullfilename')

        # get the best guess at the header row
        if len(full_filename) > 0:
            df = pd.read_excel( full_filename, header=None)
            # remove columns with all null values 
            df.dropna(axis=1, how='all', inplace=True)
            # remove rows with all null values 
            df.dropna(axis=0, how='all', inplace=True)
           # get_bestguess_table_header => returns a dictionary
            best_guess = get_bestguess_table_header(table_as_list=df.head(50).values.tolist())

            # Convert row values to list as original text of row 
            row_text = df.apply(lambda row: str(list(row.values)), axis=1)
            # Add the filename and row as text to the DataFrame
            df['rowtext'] = row_text # add as last row 
            del row_text

            # Add the filename as first column to the DataFrame
            df.insert(0, 'filename', file_data.get('filename'))

            # update the column names 
            df.columns = best_guess.get('header_guess')
            # replace all null values with a string 
            df.fillna('empty', inplace=True)
            # ignore all the rows above the header row
            df = df.iloc[best_guess.get('header_rownum') + 1:]

            table_header = {'filename': file_data.get('filename'),
                            'numcols' : len(df.columns),
                            'header_rownum':best_guess.get('header_rownum'),
                            'header_guess':best_guess.get('header_guess'), 
                            'header_raw':best_guess.get('header_raw'), 
                            'header_node_text':best_guess.get('header_node_text'),
                            'plantname_in_header':best_guess.get('plantname_in_header')}
            
            return df, table_header
        else:
            print('No table found with fitz parse by grid or text ')
            return None, None 
    except:
            pass    

def save_table_to_session(file_data, file_table, search_headers):
    """
    Updates or creates entries in the Tables and TableData session based on provided data.
    Does not commit objects -  db.session.commit() required 

    Parameters:
    - file_data: A dictionary containing 'filename', 'vendor', and 'filetoken' keys.
    - file_table: A pandas DataFrame object that represents the table data to be saved.
    - search_headers: A list of headers used for search indexing in the database.

    Returns:
    - A dictionary indicating the operation status for both 'table' and 'data' entries.
      Returns None if an exception occurs during database operations.
    """
    
    # Initialize a dictionary to keep track of the operation status
    status = {'table': None, 'data': None}

    try:
        # Retrieve an existing Tables entry or None if it doesn't exist
        existing_table = Tables.query.filter_by(
            file_name=file_data.get("filename"),
            vendor=file_data.get("vendor"),
            table_name=file_data.get("filetoken")
        ).first()

        # Update existing Tables entry or create a new one
        if existing_table:
            existing_table.vendor = file_data.get("vendor")
            existing_table.table_name = file_data.get("filetoken")
            status['table'] = 'Updated'
        else:
            new_table = Tables(
                vendor=file_data.get("vendor"), 
                file_name=file_data.get("filename"), 
                table_name=file_data.get("filetoken")
            )
            db.session.add(new_table)
            status['table'] = 'New'

        # Convert the file_table DataFrame to a JSON string for storage
        df_json = file_table.to_json(orient='split')

        # Retrieve an existing TableData entry or None if it doesn't exist
        existing_table_data = TableData.query.filter_by(table_name=file_data.get("filetoken")).first()

        # Update existing TableData entry or create a new one
        if existing_table_data:
            existing_table_data.search_columns = search_headers
            existing_table_data.row_count = len(file_table)
            existing_table_data.df = df_json
            status['data'] = 'Updated'
        else:
            new_table_data = TableData(
                table_name=file_data.get("filetoken"), 
                search_columns=search_headers,
                row_count=len(file_table), 
                df=df_json
            )
            db.session.add(new_table_data)
            status['data'] = 'New'

    except SQLAlchemyError as e:
        logging.error(f"Database operation failed: {e}")
        db.session.rollback()
        return None

    return status

def save_all_file_tables_in_dir(dirpath:str, use_dropbox = False):
    '''
    Top level function to create tables and save to db
    Automatically identifies unique table headings 
    '''
    try:
        load_dotenv()

        # NOTE: do NOT deploy with your key hardcoded
        tmp = os.getenv('OPENAI_API_KEY')
        os.environ["OPENAI_API_KEY"] = tmp

        # define embedding model 
        service_context = ServiceContext.from_defaults(embed_model="local")

        # get the files 
        if use_dropbox:
            dropbox_access_token = get_dropbox_accesstoken_from_refreshtoken
            dbx = dropbox.Dropbox(dropbox_access_token)
            dropbox_filenames = get_dropbox_filenames(dbx,startDir = '/Garden Centre Ordering Forms/OrderForms')
            filenames = [f for f in dropbox_filenames.keys() if allowed_file(f.split('/')[-1])]
        else:
            os_filenames = get_filenames_in_directory(dirpath)
            filenames = [f for f in os_filenames.keys() if allowed_file(f.split('/')[-1])]

        # yield the status
        logging.info (f'Found {len(filenames)} files in {dirpath} at {datetime.now():%b %d %I:%M %p}:')
        yield f'Found {len(filenames)} files in {dirpath} at {datetime.now():%b %d %I:%M %p}:'
        for tmp in filenames:
            yield('  ' + str(tmp))

        # loop the files, and extract tables  
        # for filename in filenames:
        # for full_filename in [filenames[0]]:
        for full_filename in filenames:
            # get the dirpath, filename and file type
            file_data = parse_fullfilename(full_filename = full_filename)

            # yield the status
            logging.info(f'{file_data.get("vendor")}: {file_data.get("filename")} at {datetime.now():%b %d %I:%M %p}')
            yield ' '
            yield f'{file_data.get("vendor")}: {file_data.get("filename")} at {datetime.now():%b %d %I:%M %p}'

            # create a well formed key to hash for db primary key 
            if file_data.get("filename"):
                filetoken = secure_filename(file_data.get("filename").split('.')[0])
            else:
                filetoken = None 
                logging.info(f'Failed filetoken extract: {file_data.get("fullfilename")}')

            # branch pdf vs. xlsx files 
            if file_data.get('filetype').lower() == 'pdf':
                # extract tables as dataframes - for local files 
                file_table, table_header = get_file_table_pdf(file_data)

            elif file_data.get('filetype').lower() in ['xls', 'xlsx']:
                # extract tables as dataframes - for local files 
                file_table, table_header = get_file_table_xls(file_data)
            
            # remove columns with all null values 
            file_table.dropna(axis=1, how='all', inplace=True)
            # remove rows with all null values 
            file_table.dropna(axis=0, how='all', inplace=True)

            # guess the columns to search for plant names 
            search_headers = get_bestguess_table_search_columns(file_table)

            # yield the status
            logging.info(f"Created {filetoken} table at {datetime.now():%b %d %I:%M %p}/nHeader: {table_header['header_guess']}")
            yield f"Created {filetoken} table at {datetime.now():%b %d %I:%M %p}"
            yield f"From: {table_header.get('header_raw')}"
            yield f"Table Header Guess: {table_header.get('header_guess')}"
            yield f"Search Header Guess: {search_headers} "

            # save the table to the session 
            status = save_table_to_session(file_data, file_table, search_headers )

        # Commit the session to save changes to the database
        db.session.commit()

    except SQLAlchemyError as e:
        logging.error(f"Database operation failed: {e}")
        db.session.rollback()

def background_task(app, dirpath, user_id):
    """
    Performs a background task to update file tables in a specified directory for a given user.

    Parameters:
    - app: The Flask application context.
    - dirpath: The directory path containing files to update.
    - user_id: The ID of the user for whom the updates are performed.

    The function updates the status of file processing in the database and logs progress.
    """
    with app.app_context():
        try:
            # Check or create task completion flag for the user
            check_thread = ThreadComplete.query.get(user_id)
            if check_thread:
                check_thread.task_complete = False
            else:
                new_thread = ThreadComplete(id=user_id, task_complete=False)
                db.session.add(new_thread)
            db.session.commit()

            # Initialize or update user data
            user = UserData.query.get(user_id)
            if not user:
                logging.error(f"User {user_id} not found.")
                return
            user_data = user.data if user.data else {}
            user_data['update_status'] = []

            # Process each file in the directory
            for update in save_all_file_tables_in_dir(dirpath, use_dropbox=False):
                logging.info(f"Background Update: {update}")
                user_data['update_status'].append(update)
                message_queue.put(update)

            # Save the updated user data
            user.data = user_data
            db.session.merge(user)
            db.session.commit()

            # Signal task completion
            completion_message = f"Completed Inventory Update at {datetime.now():%b %d %I:%M %p}"
            logging.info(completion_message)
            message_queue.put(completion_message)

            # Update task completion status
            if check_thread:
                check_thread.task_complete = True
                initStatus = f'Inventory last refreshed at {datetime.now():%b %d %I:%M %p}'
                user.data = {'status': initStatus}
                db.session.merge(user)
                db.session.commit()

            message_queue.put(None)

        except SQLAlchemyError as e:
            db.session.rollback()
            logging.error(f"Database operation failed: {e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
