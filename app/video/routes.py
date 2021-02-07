
import os
from datetime import datetime
from flask import render_template, redirect, url_for, flash, request, current_app, jsonify
from flask_login import current_user, login_required
from sqlalchemy import func
from glob import glob
import subprocess


from app import db
from app.video import bp
from app.text_summarization.summarize import get_summary
from app.video.forms import AddVideoForm
from app.decorators import role_required
from app.models import Classroom, User, Role, Video


@bp.route('/add', methods=['GET', 'POST'])
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
        upload_path = os.path.join(current_app.config['UPLOAD_PATH'], file_name)
        open(upload_path, 'wb').write(data)
        
        
        audio_path = os.path.join(current_app.config['AUDIO_PATH'], 'audio.wav')
        command = "ffmpeg -i " + upload_path + " -ab 160k -ac 2 -ar 44100 -vn -y " + audio_path
        subprocess.call(command, shell=True)
        
        
        model = current_app.config['model']
        utils = current_app.config['utils']
        device = current_app.config['device']
        args = current_app.config['args']
        predictor = current_app.config['predictor']
        segmenter = current_app.config['segmenter']
        decoder = current_app.config['decoder']
        (read_batch, split_into_batches, read_audio, prepare_model_input) = utils 

        test_files = glob(audio_path)
        batches = split_into_batches(test_files, batch_size=10)
        input = prepare_model_input(read_batch(batches[0]),
                                    device=device)

        output = model(input)
        for example in output:
            data = decoder(example.cpu())
            summary, _ = get_summary(args, data, device, predictor)
        
            
        classroom = Classroom.query.filter_by(id=int(form.classname.data)).first_or_404()
        video = Video(file = file_name, name = form.name.data, creator_id = current_user.id, original_text = data, summary = summary)
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


@bp.route('/', methods=['GET', 'POST'])
@login_required
def list_videos():
    user_id = current_user.id
    
    _role = Role.query.filter_by(id=current_user.role_id).first()
    if _role.name == 'Student':
        classes = User.query.filter_by(id = current_user.id).first().classroom.all()
    else:
        classes = Classroom.query.filter_by(creator_id = current_user.id).all()
        
    videos = []
    for classroom in classes:
        videos.extend(classroom.video.all())
    
    class_ids = []
    class_names = []
    for video in videos:
        classroom = video.classes.first()
        class_ids.append(classroom.id)
        class_names.append(classroom.name)
        
    return render_template('video/list_videos.html', videos = videos, class_ids = class_ids, 
                           class_names = class_names)    