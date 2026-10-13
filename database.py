import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

DB_NAME = Path(__file__).resolve().with_name("tickets.db")

def init_db():
    """Initializes the SQLite database table for storing ticket logs."""
    with closing(sqlite3.connect(DB_NAME)) as conn:
        with conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ticket_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    description TEXT NOT NULL,
                    predicted_category TEXT NOT NULL,
                    category_confidence REAL NOT NULL,
                    predicted_priority TEXT NOT NULL,
                    priority_confidence REAL NOT NULL,
                    corrected_category TEXT,
                    corrected_priority TEXT,
                    corrected_at TEXT
                )
            ''')
            columns = {row[1] for row in conn.execute("PRAGMA table_info(ticket_logs)")}
            for column in ("corrected_category", "corrected_priority", "corrected_at"):
                if column not in columns:
                    conn.execute(f"ALTER TABLE ticket_logs ADD COLUMN {column} TEXT")

def log_ticket(description, category, cat_conf, priority, prio_conf):
    """Saves a classified ticket record into the database."""
    with closing(sqlite3.connect(DB_NAME)) as conn:
        with conn:
            cursor = conn.execute('''
                INSERT INTO ticket_logs
                (timestamp, description, predicted_category, category_confidence, predicted_priority, priority_confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), description, category, cat_conf, priority, prio_conf))
            return cursor.lastrowid

def save_correction(ticket_id, category, priority):
    """Stores agent-confirmed category and priority labels for a ticket."""
    with closing(sqlite3.connect(DB_NAME)) as conn:
        with conn:
            cursor = conn.execute('''
                UPDATE ticket_logs
                SET corrected_category = ?, corrected_priority = ?, corrected_at = ?
                WHERE id = ?
            ''', (category, priority, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ticket_id))
            return cursor.rowcount == 1

def get_corrected_tickets():
    """Returns ticket descriptions with complete agent-confirmed labels."""
    with closing(sqlite3.connect(DB_NAME)) as conn:
        return conn.execute('''
            SELECT description, corrected_category, corrected_priority
            FROM ticket_logs
            WHERE corrected_category IS NOT NULL AND corrected_priority IS NOT NULL
        ''').fetchall()

def get_all_tickets():
    """Retrieves all logged tickets from the database."""
    with closing(sqlite3.connect(DB_NAME)) as conn:
        return conn.execute('''
            SELECT id, timestamp, description, predicted_category, category_confidence,
                   predicted_priority, priority_confidence, corrected_category,
                   corrected_priority, corrected_at
            FROM ticket_logs
            ORDER BY id DESC
        ''').fetchall()