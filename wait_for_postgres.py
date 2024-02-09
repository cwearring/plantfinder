import psycopg2
import os
import time

database_url = os.environ['DATABASE_URL']

def wait_for_postgres():
    while True:
        try:
            conn = psycopg2.connect(database_url)
            conn.close()
            print("PostgreSQL is up and running.")
            break
        except psycopg2.OperationalError:
            print("PostgreSQL is unavailable, waiting...")
            time.sleep(1)

if __name__ == "__main__":
    wait_for_postgres()
