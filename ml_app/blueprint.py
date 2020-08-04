import logging
import os

from flask import Blueprint, render_template, request, g, url_for, flash, session, redirect
from sqlalchemy.exc import IntegrityError

from flask_paginate import Pagination, get_page_parameter
from flask_app import xrays
from flask_app.auth import login_required
from ml_app.forms import ImageUploadForm, ImageDetectForm
from ml_app.models.uploads import UserUploads, DieseaseType
from ml_app.nn_architecture import load_and_predict

logger = logging.getLogger(__file__)


ml_bp = Blueprint('ml_app', __name__, url_prefix='/ml')


@ml_bp.route('/upload', methods=["GET", "POST"])
def upload_image():
    logger.info("User requested to upload the xray %s" % g.user)
    form = ImageUploadForm()
    if form.validate_on_submit():
        filename = xrays.save(form.xray.data)
        diesease = form.diesease.data
        if diesease  == '3':
            diesease = DieseaseType.NORMAL
        if diesease =='2':
            diesease = DieseaseType.Pneumonia
        if diesease == '1':
            diesease = DieseaseType.COVID19
        try:
            user_upload =  UserUploads(file=filename, diesease=DieseaseType(int(form.diesease.data)), user=g.user)
            user_upload.save()
        except IntegrityError as e:
            flash("File with name %s Already uploaded " % filename)
        flash('File Uploaded Successfuly1', category='sucess')
        return render_template('ml_app/xray-upload.html', form=form)
    return render_template('ml_app/xray-upload.html', form=form)



@ml_bp.route('/detect', methods=["GET", "POST"])
def detect():
    logger.info("User requested to detect %s" % g.user)
    session.pop('positive', None)
    form = ImageDetectForm()
    if form.validate_on_submit():
        filename = xrays.save(form.xray.data)
        xray_path = xrays.path(filename=filename)
        prediction = load_and_predict(xray_path)
        positive = prediction[0]
        diesease = 1
        try:
            user_upload = UserUploads(file=filename, diesease=diesease, user=g.user)
            user_upload.save()
        except IntegrityError as e:
            # flash("File with name %s Already uploaded " % filename)
            pass
        session['file_url'] = xrays.url(filename=filename)
        session['positive'] = positive
        return redirect(url_for('ml_app.results'))
    return render_template('ml_app/xray-detect.html', form=form)


@ml_bp.route('/results', methods=["GET"])
def results():
    logger.info("User requested to detect %s" % g.user)
    filename = session['file_url']
    positive = session['positive']
    total = sum(positive)
    positive[0] = (positive[0]/total)*100
    positive[1] = (positive[1]/total)*100
    return render_template('ml_app/results.html', file_url=filename, positive=positive)





@ml_bp.route('/my_uploads', methods=["GET"])
@login_required
def my_uploads():

    logger.info("User requested uploads  %s" % g.user)
    search = False
    q = request.args.get('q')
    if q:
        search = True
    query =  UserUploads.query.filter_by(user=g.user).order_by(UserUploads.id)
    page = request.args.get(get_page_parameter(), type=int, default=1)

    user_uploads = query.limit(10).offset((page-1) * 10)
    pagination = Pagination(page=page, per_page=10, total=query.count(), css_framework='bootstrap', search=search, record_name='users')
    return render_template('ml_app/my-uploads.html', uploads=user_uploads, pagination=pagination)


@ml_bp.route('/my_uploads/<int:xray_id>/delete', methods=["GET"])
@login_required
def delete_upload(xray_id):

    logger.info("User requested to delete the uploads  %s" % g.user)
    record =  UserUploads.query.get(xray_id).delete()
    flash(f"File {xray_id} has been removed")
    return redirect(url_for('ml_app.my_uploads'))


@ml_bp.route('/my_uploads/<int:xray_id>/approve', methods=["GET"])
@login_required
def approve_upload(xray_id):
    logger.info("User requested to delete the uploads  %s" % g.user)
    record =  UserUploads.query.get(xray_id)
    record.is_approved = True
    record.save()
    flash(f"File {xray_id} has been Approved")
    return redirect(url_for('ml_app.my_uploads'))