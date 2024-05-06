import sqlparse
import json
from pathlib import Path
import re
import pandas as pd
import sqlite3

SQL_DATA_TYPES = ['', 'varchar(15)', 'VARCHAR(15)', 'VARCHAR(64)', 'decimal(4,0)', 'DATE', 'timestamp', 'decimal(10,0)', 'text', 'char(3)', 'char(2)', 'TINYINT UNSIGNED', 'Date', 'DECIMAL(19,4)', 'numeric(8,2)', 'varchar(6)', 'VARCHAR(220)', 'char(4)', 'MEDIUMINT UNSIGNED', 'DECIMAL(5,2)', 'varchar(3)', 'CHAR(10)', 'float', 'VARCHAR(250)', 'CHAR(255)', 'DECIMAL(4,2)', 'character varchar(4)', 'VARCHAR(2)', 'date', 'VARCHAR(45)', 'varchar(50)', 'varchar(7)', 'number(10)', 'char(140)', 'VARCHAR(80)', 'number(10,2)', 'BIT', 'DOUBLE', 'VARCHAR(200)', 'SMALLINT UNSIGNED', 'varchar2(20)', 'float(4,1)', 'varchar(20)', 'decimal(10,2)', 'numeric(4,0)', 'numeric', 'varchar(80)', 'char(35)', 'DECIMAL', 'VARCHAR(255)', 'number(9,0)', 'varchar(2)', 'VARCHAR(5)', 'char(26)', 'numeric(3,0)', 'varchar2(30)', 'varchar(25)', 'varchar(40)', 'VARCHAR(50)', 'NUMERIC(10,2)', 'SMALLINT', 'VARCHAR(3)', 'varchar(160)', 'VARCHAR(4)', 'VARCHAR(288)', 'VARCHAR(32)', 'TEXT', 'char(30)', 'VARCHAR(1)', 'bigint(20)', 'VARCHAR(120)', 'varchar(12)', 'float(8)', 'varchar2(50)', 'DECIMAL(20,4)', 'VARCHAR(1024)', 'BLOB', 'VARCHAR(60)', 'varchar(255)', 'REAL', 'VARCHAR(25)', 'CHAR(1)', 'VARCHAR(20)', 'VARCHAR(10)', 'varchar(24)', 'varchar(30)', 'char(52)', 'char(60)', 'varchar(100)', 'varchar(60)', 'FLOAT', 'char(45)', 'integer', 'VARCHAR(100)', 'real', 'int', 'varchar(10)', 'BOOLEAN', 'float(10,2)', 'varchar(120)', 'varchar(128)', 'VARCHAR(13)', 'Char(4)', 'Char(50)', 'decimal(6,0)', 'VARCHAR(24)', 'numeric(2,0)', 'VARCHAR(11)', 'varchar(35)', 'CHAR(20)', 'VARCHAR(12)', 'VARCHAR(16)', 'VARCHAR(30)', 'varchar(1)', 'YEAR', 'BIGINT', 'number(7,2)', 'INTEGER', 'decimal(8,2)', 'numeric(5,0)', 'int(11)', 'varchar(18)', 'NUMERIC', 'VARCHAR(7)', 'float(3,1)', 'varchar(300)', 'varchar(8)', 'VARCHAR(70)', 'varchar(5)', 'char(20)', 'varchar(220)', 'TIMESTAMP', 'varchar(4)', 'numeric(2)', 'numeric(12,2)', 'varchar(200)', 'varchar(70)', 'decimal(2,2)', 'VARCHAR(40)', 'datetime', 'Text', 'INT', 'number(6,0)', 'CHAR(5)', 'char(1)', 'character varchar(3)', 'DATETIME', 'CHAR(15)', 'bool', 'number(4,0)', 'double', 'VARCHAR(160)', 'bigint', 'varchar2(10)', 'Char(30)', 'decimal(5,0)']

ROOT_PATH = Path("/raid/infolab/sarvam/elk_home_sarvam/pointer_decoding")
SPIDER_PATH = Path('/raid/infolab/sarvam/elk_home_sarvam/spider/')
SPIDER_DEV_PATH = SPIDER_PATH / 'processed' / 'dev_gold.json'
DATABASES_PATH = SPIDER_PATH / 'database'
SPIDER_PTR_PATH = Path("/raid/infolab/sarvam/elk_home_sarvam/pointer_decoding/data/spider/")


def get_table_schema(db_id, with_prefix=False, with_variable_tags=False):
    """
    Get schema of the database with db_id from the sqlite file

    Parameters:
    -----------
    db_id: str
        Database ID
    with_prefix: bool
        If True, return schema with prefix
    with_variable_tags: bool
        If True, return schema with variable annotation tags

    Returns:
    --------
        str: Schema of the database
    """
    if with_prefix:
        t_prefix = " "
        c_prefix = " "
    else:
        t_prefix = ""
        c_prefix = ""

    if with_variable_tags:
        table_var_begin = "<variable type='table' id='{}'>"
        table_var_end = "</variable>"
        col_var_begin = "<variable type='column' id='{}'>"
        col_var_end = "</variable>"
    else:
        table_var_begin = ""
        table_var_end = ""
        col_var_begin = ""
        col_var_end = ""

    sqlite_file_path = DATABASES_PATH / db_id / (db_id + '.sqlite')

    conn = sqlite3.connect(sqlite_file_path)
    cursor = conn.cursor()

    # Get the list of tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    table_info = []
    cid = 1
    # Fetch and print the schema for each table
    for tid, table in enumerate(tables):
        table_name = table[0]

        # Fetch column information for the table
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()

        cols = []
        # Collect column details
        for _, column in enumerate(columns):
            column_name, data_type, _, _, is_primary_key = column[1:]
            primary_key_constraint = " PRIMARY KEY" if is_primary_key else ""
            cols.append(
                f"  {col_var_begin.format(cid)}\"{c_prefix + column_name}\"{col_var_end} {data_type}{primary_key_constraint}")
            cid += 1
            # data_types.add(data_type)

        table_info.append(
            f"CREATE TABLE {table_var_begin.format(tid + 1)}\"{t_prefix + table_name}\"{table_var_end}" + "\n(" + ",\n".join(
                cols) + "\n);")

    cursor.close()
    conn.close()

    return "\n\n".join(table_info)


def get_schema_elements_from_db_id(db_id):
    """
    Get schema elements from db_id

    Parameters:
    -----------
    db_id: str
        Database ID

    Returns:
    --------
    list: List of schema elements

    Example:
    --------
    >>> cols, tables = get_schema_elements_from_db_id("concert_singer")
    """
    schema = get_table_schema(db_id)
    data_types = "|".join(SQL_DATA_TYPES)
    data_types = data_types.replace("(", "\(").replace(")", "\)")
    col_regex = fr'^\(?\s\s\"([a-zA-Z_ ]+)\" \b({data_types})\b'
    table_regex = fr'^CREATE TABLE \"([a-zA-Z_ ]+)\"'

    cols = re.findall(col_regex, schema, re.MULTILINE)
    tables = re.findall(table_regex, schema, re.MULTILINE)

    cols = [i[0] for i in cols]

    return (cols, tables)


def check_schema_items(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    problem_counter = 0
    literal_counter = 0
    foo = []
    bar = []


    for x, i in enumerate(data):
        db_id = i['db_id']
        (cols,tabs) = get_schema_elements_from_db_id(db_id)
        gold_sql = i['SQL'].lower()

        problem_counter = 0
        literal_counter = 0

        for j in list(set(cols)):
            if j.lower() in gold_sql:
                literal_counter += 1
        
        for j in list(set(tabs)):
            if j.lower() in gold_sql:
                literal_counter += 1
        
        
        foo.append(literal_counter)

    print(pd.Series(foo).describe())


            

check_schema_items("/raid/infolab/sarvam/elk_home_sarvam/pointer_decoding/data/spider/dev_gold.json")

