"""
Database operations for the reno package.

This module provides functions for interacting with the IRISE database,
including connection management, query execution, and schema inspection.
"""

import os
import sqlite3
import shutil
import subprocess
from urllib.error import URLError
from urllib.request import urlretrieve
from contextlib import contextmanager
from typing import List, Optional, Tuple, Any
from source.utils import FilePathBuilder

IRISE_DB_URL = ("https://zenodo.org/records/15499551/files/"
                "irise38.sqlite3?download=1")


def _download_with_curl(source_url: str, target_path: str) -> bool:
    """
    Download using curl, which is usually robust on macOS.

    Args:
        source_url: Remote URL to download
        target_path: Local destination file path

    Returns:
        bool: True when curl download succeeds, False otherwise.
    """
    curl_path = shutil.which("curl")
    if curl_path is None:
        return False

    command = [
        curl_path,
        "--fail",
        "--location",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "--output",
        target_path,
        source_url
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        if stderr:
            print(f"curl download failed: {stderr}")
        else:
            print("curl download failed.")
        return False


def _download_irise_database(target_path: str) -> bool:
    """
    Download the IRISE SQLite database from Zenodo.

    Args:
        target_path: Local destination path for the SQLite file

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    folder = os.path.dirname(target_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    print("IRISE database not found. Downloading from Zenodo...")
    print(f"Source: {IRISE_DB_URL}")
    print("This can take a while (~430 MB).")

    if _download_with_curl(IRISE_DB_URL, target_path):
        print(f"Database downloaded successfully: {target_path}")
        return True

    try:
        urlretrieve(IRISE_DB_URL, target_path)
    except (URLError, OSError) as exc:
        print(f"Failed to download IRISE database: {exc}")
        return False

    print(f"Database downloaded successfully: {target_path}")
    return True


@contextmanager
def get_db_connection():
    """
    Context manager for database connections.

    This function provides a context manager for SQLite database connections,
    ensuring proper connection handling and cleanup.

    Yields:
        sqlite3.Connection: A connection to the IRISE database

    Example:
        >>> with get_db_connection() as conn:
        ...     cursor = conn.cursor()
        ...     cursor.execute("SELECT * FROM some_table")
    """
    db_path = FilePathBuilder().get_irise_db_path()
    if not os.path.exists(db_path):
        downloaded = _download_irise_database(db_path)
        if not downloaded:
            print("Could not obtain IRISE database automatically.")
            print("Please download it manually into the data folder:")
            print(IRISE_DB_URL)
            exit()

    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def execute_query(query: str, params: Optional[tuple] = None
                  ) -> List[Tuple[Any, ...]]:
    """
    Execute a query and return results.

    Args:
        query: SQL query string to execute
        params: Optional tuple of parameters for the query

    Returns:
        List[Tuple[Any, ...]]: Query results as a list of tuples

    Example:
        >>> results = execute_query("SELECT * FROM houses WHERE id = ?", (1,))
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        return cursor.fetchall()


def get_table_schema(table_name: str) -> List[Tuple[Any, ...]]:
    """
    Get the schema of a table.

    Args:
        table_name: Name of the table to inspect

    Returns:
        List[Tuple[Any, ...]]: List of tuples containing column information
            (cid, name, type, notnull, dflt_value, pk)

    Example:
        >>> schema = get_table_schema("houses")
        >>> for col_info in schema:
        ...     print(f"Column: {col_info[1]}, Type: {col_info[2]}")
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        return cursor.fetchall()


def print_table_schema(table_name: str):
    """
    Print the schema of a table in a human-readable format.

    Args:
        table_name: Name of the table to inspect

    Example:
        >>> print_table_schema("houses")
        houses table schema:
        Column: id, Type: INTEGER
        Column: name, Type: TEXT
        ...
    """
    schema = get_table_schema(table_name)
    print(f"{table_name} table schema:")
    for column in schema:
        print(f"Column: {column[1]}, Type: {column[2]}")
