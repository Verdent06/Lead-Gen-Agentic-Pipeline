import psycopg2
from pgvector.psycopg2 import register_vector

DB_PARAMS = {
    "dbname": "deal_sourcing_db",
    "user": "vedant",
    "password": "rootpassword",
    "host": "localhost",
    "port": "5432"
}

def test_connection():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        
        register_vector(conn)
        
        print("SUCCESS: Connected to raw PostgreSQL and registered pgvector.")
        conn.close()
    except Exception as e:
        print(f"FATAL: Database connection failed. Error: {e}")

if __name__ == "__main__":
    test_connection()