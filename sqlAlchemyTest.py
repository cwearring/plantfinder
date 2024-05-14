from sqlalchemy import create_engine, text 


# Replace 'your_connection_string' with your actual SQLAlchemy connection string
engine = create_engine('postgresql+psycopg2://postgres:cwearring@localhost:5432/postgres')

try:
    connection = engine.connect()
    result = connection.execute(text('SELECT 1'))
    print("Connection is up and valid.")
except Exception as e:
    print("Connection failed:", str(e))
finally:
    if 'connection' in locals() and connection:
        connection.close()