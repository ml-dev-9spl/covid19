from flask_uploads import UploadSet, IMAGES
from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import StringField, BooleanField, FileField, SelectField

from flask_app import xrays

DISEASE_CHOICES = [('1', 'COVID19'), ('2', 'Pneumonia')]


class ImageUploadForm(FlaskForm):
    xray = FileField('Xray Image', validators=[ FileRequired(),  FileAllowed(xrays, 'Images only!')])
    diesease =  SelectField('Disease', choices=DISEASE_CHOICES)

class ImageDetectForm(FlaskForm):
    xray = FileField('Xray Image', validators=[ FileRequired(),  FileAllowed(xrays, 'Images only!')])