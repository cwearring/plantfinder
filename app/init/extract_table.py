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
logging.basicConfig(level=logging.INFO)

# LlmaIndex manages data and embeddings 
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor

# lower level functions for creating nodes 
from llama_index.schema import TextNode

# modules to create an http server with API
from flask import Flask, request, jsonify, current_app
from flask_login import current_user
from flask_sqlalchemy import SQLAlchemy

# https://docs.sqlalchemy.org/en/20/core/engines.html
from sqlalchemy import  Column, inspect

# queues for streaming updates during init 
from queue import Queue, Empty
import threading
# Global message queue
message_queue = Queue()

# get functions and variables created by __init__.py
from app import db, create_app
from app.models import ThreadComplete, UserData, get_user_data

def allowed_file(filename):
    '''
    CHeck filename against allowed extensions 
    '''
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'xls', 'xlsx'}

    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_hash(text):
    hash_object = hashlib.sha256(text.encode())
    return hash_object.hexdigest()

def is_file_dropbox(dropboxMeta):
    '''
    Check a drop box object id and test if it is a file
    Returns True if object is file, False otherwise
    '''
    return isinstance(dropboxMeta,dropbox.files.FileMetadata)

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
                if entry.is_file():
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

def best_header_word_match(header_word, header_words):
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

def get_bestguess_table_header(table_as_list = None):
    '''
    Uses embedding and LLM to guess the best header row
    : returns a dictionary
        'header_rownum' : header_rownum,
        'header_node_text' : header_node_text,
        'header_raw' : header_raw,
        'p_key' : p_key, a variable with the name of the sqlalchemy table primary key 
        'header_guess' : header_guess
    '''
    try:
        # get the best guess at the header row
        if len(table_as_list) > 0:
            # guess the number of columns from the most frequent list length 
            numcols = most_frequent_integer([len(x) for x in table_as_list])
            # find the first list with the right number of columns 
            for n,row in enumerate(table_as_list):
                if len(row) == numcols:
                    header_rownum_guess = n
                    break

            # create a list of text nodes with one node per row in tmp 
            table_nodes = [TextNode(text=f"'{t}'", id_=n) for n,t in enumerate(table_as_list)]
            # table_nodes = [TextNode(t, id_=n) for n,t in enumerate(tmp)]
            table_index = VectorStoreIndex(table_nodes)

            # create the query engine with custom config to return just one node
            # https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/root.html#query-engine 
            query_engine = table_index.as_query_engine(
                similarity_top_k=1,
                vector_store_query_mode="default",
                alpha=None,
                doc_ids=None,
            )
            # get the gpt's best guess
            oneshotprompt=f"Retrieve column headings as a python list for a product price sheet with {numcols} columns."

            multishotprompt=f"""Retrieve a column heading for a product price sheet with {numcols} columns.
            Return an existing row. Do not make up rows.

            5 column example ['Name', 'Latin Name', 'Price', 'Available Qty', 'Order Qty']
            8 column example [ "Product","SIZE1","SIZE2","PRICE", "AVL",  "COMMENTS", "ORDER", "Total"]
            7 column example ["Category", "WH", "Code", "Botantical Name", "size", "Price", "Available"]
            """
            response = query_engine.query(multishotprompt)

            # get info from the header row
            header_rownum = int(response.source_nodes[0].id_)
            header_node_text = response.source_nodes[0].text
            header_raw = table_as_list[header_rownum]
            p_key = 'id_hash'
            
            # correct for failure of table header guess 
            if len(header_raw) == numcols:
                header_guess = [p_key] + [str(x).replace(' ','').replace('\n','_').replace('(','_').replace(')','')
                                        for x in header_raw] + ['TableName', 'Text'] 
            else:
                header_guess = [p_key] + [f'col_guess_{n}' for n in range(0,numcols)]+ ['TableName', 'Text']

            best_guess={    
            'header_rownum' : header_rownum,
            'header_node_text' : header_node_text,
            'header_raw' : header_raw,
            'p_key' : p_key,
            'header_guess' : header_guess
            }
            
            return best_guess
        else:
            return None
    except:
        return None
        pass

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
    # define a list of header words from the docs 
    header_words = ['Product', 'Variety', 'Size', 'Colour', 'Order Qty', 'Cost', 'Description', 'Code', 'Name',\
                'Category','Your Price', 'Price', 'Status', 'Zone', 'Catalog $', 'Pots/Tray', 'Amount',\
                'WH', 'Botanical Name', 'E-LINE', 'Available','Order', 'Total', 'PIN', 'UPC','Latin Name',\
                'Available Units','QTY', 'Notes','Avail QTY','Order Qty','Plant Name','Common Name','Sale Price',\
                'Pot Size','List','Net','Comments','AVL','Sku','Case Qty','Packaging', "Pots Ordered", 'SIZE 1', 'SIZE 2']
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

            # save the file header info
            p_key = best_guess.get('p_key')

            # merge with the table from the first page 
            table_data = [[generate_hash(f"{row}")] + row + 
                          [file_data.get('filename'), f"{row}"] for row in tmp[best_guess.get('header_rownum')+1:]]
            
            # add the tables from the rest of the pages using the same tbl_strategy 
            for pagenum in range(1,doc.page_count):
                logging.info(f"Processing page {pagenum} of {doc.page_count}")

                tbls = doc[pagenum].find_tables(vertical_strategy=tbl_strategy, horizontal_strategy=tbl_strategy)
                # exclude rows identical to the header row 
                tbl_page = [row for tbl in tbls.tables for row in tbl.extract() 
                            if row != best_guess.get('header_raw')]
                # add rows with p_key of entire row as text and add filename and text string 
                table_data.extend([[generate_hash(f"{row}")] + row + 
                                   [file_data.get('filename'), f"{row}"] for row in tbl_page])
                jnk = 0 

            # create a pandas dataframe 
            df = pd.DataFrame(table_data, columns = best_guess.get('header_guess'))
            df.set_index(p_key, inplace=True, drop=False)

            # save the file header info
            table_header = {'filename': file_data.get('filename'),
                            'numcols' : len(df.columns),
                            'header_rownum':best_guess.get('header_rownum'),
                            'header_guess':best_guess.get('header_guess'), 
                            'header_raw':best_guess.get('header_raw'), 
                            'header_node_text':best_guess.get('header_node_text')}
                        
            return df, table_header, p_key
        else:
            logging.error('No table found with fitz parse by grid or text ')
            return None, None, None    
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
           # get_bestguess_table_header returns a dictionary
            best_guess = get_bestguess_table_header(table_as_list=df.head(50).values.tolist())
            
            # the next commands insert the first column as a hashed id
            # add the tablename and original row as list as text as new columns 
            # replace the columns headings with best_guess and 
            # drop rows above and including the header row

            # Convert row values to list as original text of row 
            row_text = df.apply(lambda row: str(list(row.values)), axis=1)
            # Convert row values to list, make the list a string, and apply the hash function
            hash_values = df.apply(lambda row: generate_hash(str(list(row.values))), axis=1)
            # Insert the hash values as the first column of the DataFrame
            df.insert(loc=0, column='hash_row', value=hash_values)
            del hash_values
            # Add the filename and row as text to the DataFrame
            df['filename'] = file_data.get('filename')
            df['rowtext'] = row_text
            del row_text
            # update the column names 
            df.columns = best_guess.get('header_guess')
            # replace all null values with a string 
            df.fillna('empty', inplace=True)
            # ignore all the rows above the header row
            df = df.iloc[best_guess.get('header_rownum') + 1:]
 
            # save the file header info
            p_key = best_guess.get('p_key')

            table_header = {'filename': file_data.get('filename'),
                            'numcols' : len(df.columns),
                            'header_rownum':best_guess.get('header_rownum'),
                            'header_guess':best_guess.get('header_guess'), 
                            'header_raw':best_guess.get('header_raw'), 
                            'header_node_text':best_guess.get('header_node_text')}
            return df, table_header, p_key
        else:
            print('No table found with fitz parse by grid or text ')
            return None, None, None    
    except:
            pass    
    
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

    try:
        #with current_app.app_context():
        # Get an inspector object from the database connection
        inspector = inspect(db.engine)

        # Create ORM class from DataFrame
        DynamicClass = \
            create_class_from_df(df,
                                 class_name,
                                 p_key 
                                 )

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

def parse_fullfilename(full_filename:str = None):
    '''
    Parse a full file URI into components
    : returns a dictionary 
        'fullfilename': full_filename,
        'dirpath':dirpath, 
        'filename':filename, 
        'filetype':filetype 
    '''
    try:
        # get the filename and directory path and file type
        filetype = secure_filename(full_filename).split('.')[-1]
        filename = full_filename.split('/')[-1]
        dirpath = '/'.join(full_filename.split('/')[0:-1])    
        return {'fullfilename': full_filename,'dirpath':dirpath, 'filename':filename, 'filetype':filetype }
    except:
        return {'fullfilename':None,'dirpath':None, 'filename':None, 'filetype':None }

def save_all_file_tables_in_dir(dirpath:str, use_dropbox = False):
    '''
    Top level function to create sqlalchemy classes and save to db
    Creates new db tables in sqlalchemy db for each unique file name
    Automatically identifies unique table headings 
    '''
    load_dotenv()
    # app = current_app #     print(current_app.name)

    # NOTE: do NOT deploy with your key hardcoded
    tmp = os.getenv('OPENAI_API_KEY')
    os.environ["OPENAI_API_KEY"] = tmp
    # print(tmp)

    # define embedding model 
    service_context = ServiceContext.from_defaults(embed_model="local")

    # define the dicts to save headers and tables 
    file_table_header = {}
    file_table = {}

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
    for full_filename in filenames[4:]:
        # get the dirpath, filename and file type
        file_data = parse_fullfilename(full_filename = full_filename)

        # yield the status
        logging.info(f'Started processing {file_data["filename"]} at {datetime.now():%b %d %I:%M %p}')
        yield ' '
        yield f'Started processing {file_data["filename"]} at {datetime.now():%b %d %I:%M %p}'

        # create a well formed key to hash for db primary key 
        filetoken = secure_filename(file_data["filename"]).split('.')[0]

        # branch pdf vs. xlsx files 
        if file_data.get('filetype').lower() == 'pdf':
            # extract tables as dataframes - for local files 
            file_table, table_header, p_key = get_file_table_pdf(file_data)

        elif file_data.get('filetype').lower() in ['xls', 'xlsx']:
            # extract tables as dataframes - for local files 
            file_table, table_header, p_key = get_file_table_xls(file_data)
        
        # yield the status
        logging.info(f"Created {filetoken} table at {datetime.now():%b %d %I:%M %p}/nHeader: {table_header['header_guess']}")
        yield f"Created {filetoken} table at {datetime.now():%b %d %I:%M %p}"
        yield f"Header: {table_header['header_guess']} \n From: {table_header['header_raw']}"

        # save a sqlalchemy ORM class for each dataframe - add column = id:int as primary key
        status = save_class_in_session(df=file_table, class_name=filetoken, p_key=p_key)

        if status:
            logging.info(f"Created ORM class and db table {filetoken}")
            # yield f"Created ORM class and db table {filetoken} at {datetime.now():%b %d %I:%M %p}"
            yield f"Updated {len(file_table)} rows for {file_data['filename']} at {datetime.now():%b %d %I:%M %p}"
        else:
            logging.info(f"Hit an error save_class_in_session for {file_data['filename']}")
            yield f"Hit an error save_class_in_session for {file_data['filename']}"

    # Commit the session to save changes to the database
    db.session.commit()

def background_task(app, dirpath, user_id):
    with app.app_context():

        # create the initial value as false or update existing 
        check_thread = ThreadComplete.query.get(user_id)
        if check_thread:
            check_thread.task_complete = False
            db.session.commit()
        else:
            new_thread = ThreadComplete(id=user_id, task_complete=False)
            db.session.add(new_thread)
            db.session.commit()

        # initialize the list of updated files 
        user = UserData.query.get(user_id)
        tmptest = user.data
        tmptest['update_status'] = []

        # save the update status for each file 
        for update in save_all_file_tables_in_dir(dirpath, use_dropbox=False):
            print(f"background Update: {update}")
            # Append the update message to the 'update_status' list
            # user_data = UserData.query.get(user_id)
            tmptest['update_status'].append(update)
            message_queue.put(update)

        # Save the updated user_data after the last update 
        user = UserData.query.get(user_id)
        user.data = tmptest 
        db.session.merge(user)
        db.session.commit()

        # Signal the completion of the task
        message_queue.put(f"Completed Inventory Update at {datetime.now():%b %d %I:%M %p}")

        # Update the task as complete
        existing_thread = ThreadComplete.query.get(str(user_id))
        if existing_thread:
            existing_thread.task_complete = True
            
            # save in the database with key = user_id
            initStatus = f'Inventory last refreshed at {datetime.now():%b %d %I:%M %p}'
            user_data = UserData(id=user_id, data={'status':initStatus})
            db.session.merge(user_data)
            db.session.commit()
            
        message_queue.put(None)

