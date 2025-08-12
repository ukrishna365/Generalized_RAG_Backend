#!/usr/bin/env python3
"""
Script to clear all data from the RAG database tables
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection
DATABASE_URL = os.getenv("POSTGRES_URL", "postgresql+psycopg2://user:password@localhost/your_db")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def clear_database():
    """Clear all data from the database tables"""
    session = Session()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Starting database cleanup...")
    
    try:
        # Clear all tables in the correct order (respecting foreign keys)
        print("Clearing vector_data table...")
        session.execute(text("DELETE FROM vector_data"))
        
        print("Clearing image_data table...")
        session.execute(text("DELETE FROM image_data"))
        
        print("Clearing table_data table...")
        session.execute(text("DELETE FROM table_data"))
        
        print("Clearing text_chunk table...")
        session.execute(text("DELETE FROM text_chunk"))
        
        print("Clearing file table...")
        session.execute(text("DELETE FROM file"))
        
        # Commit the changes
        session.commit()
        
        final_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{final_timestamp}] Database cleared successfully!")
        
        # Verify tables are empty
        print("\nVerifying tables are empty:")
        tables = ['file', 'text_chunk', 'image_data', 'table_data', 'vector_data']
        for table in tables:
            result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            print(f"  {table}: {count} records")
        
    except Exception as e:
        print(f"Error clearing database: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    print("RAG Database Cleanup Tool")
    print("=" * 40)
    print("This will delete ALL data from the database.")
    
    # Ask for confirmation
    confirm = input("\nAre you sure you want to clear all data? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        clear_database()
    else:
        print("Database cleanup cancelled.") 