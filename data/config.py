"""
ProjectionAI - Database Configuration
Remote database connection settings
"""

import os
from typing import Optional

# Remote database configuration
DB_CONFIG = {
    'host': '192.168.1.23',
    'port': 5432,
    'database': 'baseball_migration_test',
    'user': 'postgres',
    'password': 'korn5676'
}

# Alternative: local database for development
# Uncomment to use local database
# DB_CONFIG = {
#     'host': 'localhost',
#     'port': 5432,
#     'database': 'projectionai',
#     'user': 'futurepr0n',
#     'password': None  # Use trust auth
# }


def get_db_config() -> dict:
    """Get database configuration"""
    return DB_CONFIG


def get_connection_string() -> str:
    """Get PostgreSQL connection string"""
    return f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
