
import os
from datetime import datetime
from flask import render_template, redirect, url_for, flash, request, current_app, jsonify
from flask_login import current_user, login_required
from sqlalchemy import func

from app import db
from app.video import bp
from app.video.forms import AddVideoForm
from app.decorators import role_required
from app.models import Classroom, User, Role, Video


@bp.route('/', methods=['GET', 'POST'])
@login_required
@role_required('Professor')
def add_video():
    form = AddVideoForm()
    form.classname.choices = [(_classroom.id, _classroom.name) for _classroom in Classroom.query.filter_by(creator_id=current_user.id).all()]
    if form.validate_on_submit():
        id = db.session.query(func.max(Video.id)).scalar()
        if id is None:
            id = 0
        id += 1
        file_name = 'file' + str(id)
        data = request.files[form.video.name].read()
        open(os.path.join(current_app.config['UPLOAD_PATH'], file_name), 'wb').write(data)
        
        classroom = Classroom.query.filter_by(id=int(form.classname.data)).first_or_404()
        video = Video(file = file_name, name = form.name.data, creator_id = current_user.id)
        video.classes.append(classroom)
        

        db.session.add(video)
        db.session.commit()
        flash('Your new video has been saved.')
        return redirect(url_for('main.index'))
    return render_template('video/add_video.html', title='Add Video', form=form)

@bp.route('/<int:classroom_id>/<int:video_id>', methods=['GET'])
@login_required
def show_video(classroom_id, video_id):
    video = Video.query.filter_by(id = video_id).first()
    classroom = Classroom.query.filter_by(id = classroom_id).first()
    return render_template('video/display_video.html', video = video, classroom = classroom)