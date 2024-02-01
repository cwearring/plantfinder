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

def generate_hash(text):
    hash_object = hashlib.sha256(text.encode())
    return hash_object.hexdigest()

def get_filenames_in_directory(directory_path):
    try:
        # List to store filenames
        filenames = []
        filepathnames = []

        # Iterate over all entries in the directory
        for entry in os.listdir(directory_path):
            # Get the full path
            full_path = os.path.join(directory_path, entry)

            # Check if it's a file and add to the list
            if os.path.isfile(full_path) and entry[0] != '.' and entry.split('.')[-1] == 'pdf':
                filenames.append(entry)
                filepathnames.append(full_path)

        return filenames, filepathnames
    
    except Exception as e:
        # Optionally, log the error here
        # log.error(f"Error in save_class_in_session: {str(e)}")

        # Return a status indicating an error occurred and include error details
        # raise ValueError(f"An error occurred in get_filenames_in_directory: {str(e)}")
    
        return None, None
    
def most_common_header(list_of_lists):
    count_dict = {}
    for lst in list_of_lists:
        # Convert list to tuple for hashing
        tuple_version = tuple(lst[1])
        if tuple_version in count_dict:
            count_dict[tuple_version] += 1
        else:
            count_dict[tuple_version] = 1

    # Find the tuple with the maximum count
    most_common = max(count_dict, default=None, key=count_dict.get)

    # Convert tuple back to list
    return list(most_common)

def most_frequent_integer(int_list):
    if not all(isinstance(x, int) for x in int_list):
        raise ValueError("The list must contain only integers")

    frequency = {}
    for num in int_list:
        if num in frequency:
            frequency[num] += 1
        else:
            frequency[num] = 1

    most_frequent = None
    max_frequency = 0

    for num, freq in frequency.items():
        if freq > max_frequency:
            most_frequent = num
            max_frequency = freq

    return most_frequent

def string_to_list(string):
    try:
        result = ast.literal_eval(string)
        if isinstance(result, list):
            return result
        else:
            raise ValueError("The evaluated expression is not a list")
    except Exception as e:
        raise ValueError(f"Error converting string to list: {e}")

def extract_text_within_brackets(input_string):
    # Define the regex pattern to find text within square brackets
    pattern = r'\[(.*?)\]'

    # Use re.findall() to find all occurrences in the string
    matches = re.findall(pattern, input_string)

    # Handling multiple matches - here, returning all of them
    return matches

def get_firstpage_tables_as_list(doc=None):

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

    return tbl_out, col_count, table_strategy

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

def get_file_table(dirpath:str, filename:str):
    '''
    Input: Directory path and filename 
    Output: Dict of header values, Dict of table valuesl, number of columns
    '''
    # define a list of header words from the docs 
    header_words = ['Product', 'Variety', 'Size', 'Colour', 'Order Qty', 'Cost', 'Description', 'Code', 'Name',\
                'Category','Your Price', 'Price', 'Status', 'Zone', 'Catalog $', 'Pots/Tray', 'Amount',\
                'WH', 'Botanical Name', 'E-LINE', 'Available','Order', 'Total', 'PIN', 'UPC','Latin Name',\
                'Available Units','QTY', 'Notes','Avail QTY','Order Qty','Plant Name','Common Name','Sale Price',\
                'Pot Size','List','Net','Comments','AVL','Sku','Case Qty','Packaging', "Pots Ordered", 'SIZE 1', 'SIZE 2']
    
    # read the doc and get the tables from the first page  
    doc = fitz.open(f'{dirpath}/{filename}')

    tmp, numcols, tbl_strategy = get_firstpage_tables_as_list(doc)

    # define embedding model 
    service_context = ServiceContext.from_defaults(embed_model="local")

    # get the best guess at the header row
    if len(tmp) > 0:
        # create a list of text nodes with one node per row in tmp 
        table_nodes = [TextNode(text=f"'{t}'", id_=n) for n,t in enumerate(tmp)]
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
        response = query_engine.query(\
        f"Retrieve column headings as a python list for a product price sheet with {numcols} columns ")

        # get info from the header row
        header_rownum = int(response.source_nodes[0].id_)
        header_node_text = response.source_nodes[0].text
        header_raw = tmp[header_rownum]
        p_key = 'id_hash'
        header_guess = [p_key] + [x.replace(' ','').replace('\n','_').replace('(','_').replace(')','')
                                  for x in header_raw] + ['TableName', 'Text'] 

        # match to header_word list 
        # header_guess = [best_header_word_match(word, header_words) for word in tmp[header_rownum]]
        
        # debug and logging 
        #print(#response.source_nodes[0].id_, '\n',
              # response.source_nodes[0].text,'\n',
              # tmp[int(response.source_nodes[0].id_)], '\n',
              # header_guess)
        
        # merge with the table from the first page 
        table_data = [[generate_hash(f"{row}")] + row + [filename, f"{row}"] for row in tmp[header_rownum+1:]]
        
        # add the tables from the rest of the pages 
        for pagenum in range(1,doc.page_count):
            tbls = doc[pagenum].find_tables(vertical_strategy=tbl_strategy, horizontal_strategy=tbl_strategy)
            tbl_page = [row for tbl in tbls.tables for row in tbl.extract() if row != header_raw]
            table_data.extend([[generate_hash(f"{row}")] + row + [filename, f"{row}"] for row in tbl_page])

        jnk=0
        # create a pandas dataframe 
        df = pd.DataFrame(table_data, columns = header_guess)
        df.set_index('id_hash', inplace=True, drop=False)

        # check for distinct values 
        # for col_nm in df.columns.tolist():
            # print(f"{col_nm}: {df[col_nm].nunique()}  of {len(df)}")

        # save the file header info
        table_header = {'filename': filename,
                        'numcols' : numcols,
                        'header_rownum':header_rownum,
                        'header_guess':header_guess, 
                        'header_raw':header_raw, 
                        'header_node_text':header_node_text}

        return df, table_header, p_key

    else:
        print('No table found with fitz parse by grid or text ')
        return None, None, None
    
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
            p_key : Column(db.CHAR(64), primary_key=True, convert_unicode=True),
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
                                 class_name.lower().replace(' ','_').replace('-','_').replace('.','_')[0:12],
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

def save_all_file_tables_in_dir(dirpath:str):

    load_dotenv()
    # app = current_app #     print(current_app.name)

    # NOTE: for local testing only, do NOT deploy with your key hardcoded
    tmp = os.getenv('OPENAI_API_KEY')
    os.environ["OPENAI_API_KEY"] = tmp
    # print(tmp)

    # define the dicts to save headers and tables 
    file_table_header = {}
    file_table = {}
    file_class = {}

    # get the files and yield the status
    filenames, filepath = get_filenames_in_directory(dirpath)
    yield f'Found {len(filenames)} files:\n'
    for tmp in filenames:
        yield('  ' + str(tmp))

# loop the files, and extract tables  
    for filename in filenames:
    # for filename in [filenames[0]]:
        print(f'\nStarted processing {filename} at {datetime.now():%b %d %I:%M %p}')
        yield ' '
        yield f'Started processing {filename} at {datetime.now():%b %d %I:%M %p}'

        # create a well formed key
        filetoken = filename.split('.')[0].replace(' ', '')

        # extract tables as dataframes 
        file_table, file_table_header, p_key = get_file_table(dirpath = dirpath, filename = filename)

        print(f"Created {filetoken} table at {datetime.now():%b %d %I:%M %p}/nHeader: {file_table_header['header_guess']}")
        yield f"Created {filetoken} table at {datetime.now():%b %d %I:%M %p}"
        yield f"Header: {file_table_header['header_guess']} \n From: {file_table_header['header_raw']}"

        # save a sqlalchemy ORM class for each dataframe - add column = id:int as primary key
        status = save_class_in_session(df=file_table, class_name=filetoken, p_key=p_key)

        if status:
            print(f"Created ORM class and db table {filetoken}")
            # yield f"Created ORM class and db table {filetoken} at {datetime.now():%b %d %I:%M %p}"
            yield f"Updated available inventory for {filename} at {datetime.now():%b %d %I:%M %p}"
        else:
            print(f"Hit an error ")
            yield f"Hit an error "

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
        for update in save_all_file_tables_in_dir(dirpath):
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
