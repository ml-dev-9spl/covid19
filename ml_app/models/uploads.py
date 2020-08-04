import enum

from flask_app import db, xrays
from flask_app.models.abc import BaseModel, MetaBaseModel


class DieseaseType(enum.IntEnum):

    COVID19 = 1
    Pneumonia = 2
    NORMAL = 3


class UserUploads(db.Model, BaseModel, metaclass=MetaBaseModel):
    """ The User model """

    __tablename__ = "user_uploads"

    id = db.Column(db.Integer, primary_key=True)
    file = db.Column(db.String(100), unique=True)
    diesease = db.Column(db.Enum(DieseaseType), default=DieseaseType.COVID19)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'),
                            nullable=False)
    user = db.relationship('User',
                               backref=db.backref('uploads', lazy=True))
    is_approved = db.Column(db.Boolean, default=False)

    def __init__(self, *args, **kwargs):
        super(UserUploads, self).__init__(*args, **kwargs)

    @property
    def get_diesease_display(self):
        return self.diesease.name

    @property
    def get_xray_url(self):
        return xrays.url(filename=self.file)