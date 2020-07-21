import logging
import os
from flask import Flask
from flask.logging import default_handler
from flask_caching import Cache
from flask_uploads import UploadSet, configure_uploads
from flask_wtf import CSRFProtect

from flask_app.db import POSTGRES, db


csrf = CSRFProtect()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'media')
xrays = UploadSet('images', tuple('jpg jpe jpeg png'.split()))

# cache settings
config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "simple", # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}
cache = Cache(config=config)

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # enable csrf protection in all the form i.e project wide
    csrf.init_app(app)
    app.config['UPLOADS_DEFAULT_DEST'] = UPLOAD_FOLDER

    # cache init
    cache.init_app(app)
    # destination for images uploads
    configure_uploads(app, (xrays,))

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    # ensure the media folder exists
    try:
        os.makedirs(app.instance_path)
        os.makedirs('media')
    except OSError:
        pass

    # add database url to connect with the postgres
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://%(user)s:%(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES
    db.init_app(app)

    from . import auth
    app.register_blueprint(auth.bp)

    from ml_app.blueprint import ml_bp
    app.register_blueprint(ml_bp)

    from . views import main_bp
    app.register_blueprint(main_bp)

    app.add_url_rule('/', endpoint='index')

    for logger in (app.logger, logging.getLogger('sqlalchemy')):
        logger.addHandler(default_handler)

    return app
