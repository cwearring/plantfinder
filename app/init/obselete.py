# holding area for unused 
def header_word_score(table:list = None):
    # define a list of header words from the docs 
    header_words = ['Product', 'Variety', 'Size', 'Colour', 'Order Qty', 'Cost', 'Description', 'Code', 'Name',\
                'Category','Your Price', 'Price', 'Status', 'Zone', 'Catalog $', 'Pots/Tray', 'Amount',\
                'WH', 'Botanical Name', 'E-LINE', 'Available','Order', 'Total', 'PIN', 'UPC','Latin Name',\
                'Available Units','QTY', 'Notes','Avail QTY','Order Qty','Plant Name','Common Name','Sale Price',\
                'Pot Size','List','Net','Comments','AVL','Sku','Case Qty','Packaging', "Pots Ordered", 'SIZE 1', 'SIZE 2']
 
    # Initial check for null string
    if not table:
        return None

    return None

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

def cell_diff(cells: list) -> list:
    """
    Calculate the element-wise differences between consecutive vectors in a list.
    
    Parameters:
    - cells: A list of lists (vectors) where each inner list contains numerical values.
    
    Returns:
    - A list of lists containing the rounded element-wise differences between consecutive vectors.
    
    Raises:
    - ValueError: If 'cells' contains less than two vectors or if any vector contains non-numeric values.
    - TypeError: If 'cells' is not a list of lists.
    """
    
    # Check if 'cells' is a list of lists
    if not all(isinstance(cell, tuple) for cell in cells):
        raise TypeError("All elements in 'cells' must be lists.")
    
    # Check if 'cells' has at least two vectors
    if len(cells) < 2:
        raise ValueError("The 'cells' list must contain at least two vectors to compute differences.")
    
    # Ensure all elements in each vector are numeric
    for vec in cells:
        if not all(isinstance(num, (int, float)) for num in vec):
            raise ValueError("All elements in each vector must be numeric.")

    def vec_diff(vec1, vec2):
        """Calculate and return the rounded element-wise difference between two vectors."""
        return [round(v1 - v2, 2) for v1, v2 in zip(vec1, vec2)]
    
    # Compute the differences between consecutive vectors
    c_diff = [vec_diff(cells[n-1], cells[n]) for n in range(1, len(cells))]

    return c_diff

def compare_absolute_values_at_index(tuples_list, index, float_val):
    """
    Compares absolute values rounded to 2 decimals to a specified index in each tuple of a list against the absolute value of a float.

    Parameters:
    - tuples_list: List of tuples containing numerical values.
    - index: The index to check in each tuple.
    - float_val: The float value to compare against.

    Returns:
    - A list of boolean values, True if the absolute value at the specified index is equal to the absolute value of the float,
      False otherwise.
      
    Raises:
    - IndexError: If the specified index is out of range for any tuple.
    """
    result = []
    for tuple_val in tuples_list:
        try:
            # Calculate the absolute value of the difference and compare it within 5% of the float_val's absolute value
            difference = abs(abs(tuple_val[index]) - abs(float_val))
            tolerance = abs(float_val) * 0.05
            result.append(difference <= tolerance)
        except IndexError:
            raise IndexError(f"Index {index} is out of range for the tuple {tuple_val}.")
    
    return result
