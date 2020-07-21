"""
Define the User model
"""
import enum

from .abc import BaseModel, MetaBaseModel
from .. import db


class UserType(enum.IntEnum):
    ADMIN = 1
    DOCTOR = 2
    USER = 3


class User(db.Model, BaseModel, metaclass=MetaBaseModel):
    """ The User model """

    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(64), unique=True, index=True)
    first_name = db.Column(db.String(300), nullable=True)
    last_name = db.Column(db.String(300), nullable=True)
    password_hash = db.Column(db.String(128), nullable=True)
    user_type= db.Column(db.Enum(UserType), default=UserType.ADMIN)

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)

    def __repr__(self):
        return '<User {}>'.format(self.id)