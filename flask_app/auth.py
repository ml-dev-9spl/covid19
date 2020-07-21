import functools
import logging

from flask import Blueprint, session, g, current_app
from flask import request, url_for, flash, render_template
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import redirect
from flask_app import db
from flask_app.forms import RegistrationForm, LoginForm
from flask_app.models.users import User
# from flask_app.db import get_db
bp = Blueprint('auth', __name__, url_prefix='/auth')

logger = logging.getLogger('flask.app')


def login_required(view):
    @functools.wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.user is None:
            return redirect(url_for('auth.login', next=request.full_path))
        return view(*args, **kwargs)
    return wrapped_view


@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = User.query.get(user_id)


@bp.route('/logout')
def logout():
    logger.info("User %3s requested session logout" % (g.user))
    session.clear()
    return redirect(url_for('main.index'))


@bp.route('/register', methods=('GET', 'POST'))
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data
        password= form.password.data
        email = form.email.data
        if User.query.filter_by(username=username).first() is not None:
            error = 'User {} is already registered.'.format(username)
            form.username.errors.append(error)
            return render_template('auth/register.html', form=form)

        password_hash = generate_password_hash(password)
        user_obj = User(
            username=username,
            password_hash=password_hash,
            email=email
        )
        db.session.add(user_obj)
        db.session.commit()
        return redirect(url_for('auth.login'))
    return render_template('auth/register.html', form=form)


@bp.route('/login', methods=('GET', 'POST'))
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        try:
            user = User.query.filter_by(username=username).first()
            hash_check = check_password_hash(user.password_hash, password)
            if hash_check:
                session.clear()
                session['user_id'] = user.id
                if request.form.get('next'):
                    return redirect(request.form.get('next'))
                return redirect(url_for('main.index'))
            form.username.errors.append('Username or Password Incorrect')
        except:
            pass
    rdrct = request.args.get('next', None)
    return render_template('auth/login.html' , next=rdrct, form=form)