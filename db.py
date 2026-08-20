# db.py

import taos
from config import (
    TDENGINE_HOST,
    TDENGINE_PORT,
    TDENGINE_USER,
    TDENGINE_PASSWORD,
    TDENGINE_DATABASE,
)


def get_conn(use_db: bool = True):
    connect_kwargs = dict(
        host=TDENGINE_HOST,
        port=TDENGINE_PORT,
        user=TDENGINE_USER,
    )
    if TDENGINE_PASSWORD:
        connect_kwargs["password"] = TDENGINE_PASSWORD

    conn = taos.connect(**connect_kwargs)
    if use_db:
        conn.execute(f"USE {TDENGINE_DATABASE}")
    return conn


def execute(sql: str):
    conn = get_conn()
    try:
        conn.execute(sql)
    finally:
        conn.close()


def query(sql: str):
    conn = get_conn()
    try:
        return conn.query(sql)
    finally:
        conn.close()
