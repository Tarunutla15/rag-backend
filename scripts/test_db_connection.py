"""
Test direct PostgreSQL connection (Supabase DB).
Uses the same env vars as the app: SUPABASE_DB_URL or SUPABASE_DB_USER/PASSWORD/HOST/PORT/NAME.
Run from backend: python -m scripts.test_db_connection
"""
import os
import sys

# Add backend to path so app.config is available
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

try:
    import psycopg2
except ImportError:
    print("Install: pip install psycopg2-binary")
    sys.exit(1)

# Prefer same vars as the app (SUPABASE_DB_*), fallback to lowercase user/password/host/port/dbname
USER = os.getenv("SUPABASE_DB_USER") or os.getenv("user")
PASSWORD = os.getenv("SUPABASE_DB_PASSWORD") or os.getenv("password")
HOST = os.getenv("SUPABASE_DB_HOST") or os.getenv("host")
PORT = os.getenv("SUPABASE_DB_PORT") or os.getenv("port", "5432")
DBNAME = os.getenv("SUPABASE_DB_NAME") or os.getenv("dbname", "postgres")
URL = os.getenv("SUPABASE_DB_URL", "").strip()

try:
    if URL:
        connection = psycopg2.connect(URL)
    elif USER and PASSWORD and HOST:
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME,
        )
    else:
        print("Set SUPABASE_DB_URL or (SUPABASE_DB_USER, SUPABASE_DB_PASSWORD, SUPABASE_DB_HOST) in .env")
        sys.exit(1)

    print("Connection successful!")
    cursor = connection.cursor()
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    print("Current time:", result)
    cursor.close()
    connection.close()
    print("Connection closed.")
except Exception as e:
    print(f"Failed to connect: {e}")
    sys.exit(1)
