import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


POSTGRES = {
    'user': 'postgres',
    'pw': 'postgres',
    'db': 'covid19',
    'host': 'localhost',
    'port': '5432',
}



