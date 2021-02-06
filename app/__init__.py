import os
from flask import Flask, request, current_app
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from config import Config

import sys
import torch

sys.path.append('/home/daniel/NLP/PreSumm/src')

from app.text_summarization.summarize import get_args, build_predictor, get_segmenter, initialize



db = SQLAlchemy()
login = LoginManager()
login.login_view = 'auth.login'
login.login_message = 'Please log in to access this page.'
bootstrap = Bootstrap()
moment = Moment()

device = torch.device('cpu')

model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', # also available 'de', 'es'
                                       device=device)




def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.config['UPLOAD_PATH'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static')
    app.config['AUDIO_PATH'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'audio')
    
    args = get_args()
    predictor = initialize(args)
    segmenter = get_segmenter()
    app.config['args'] = args
    app.config['segmenter'] = segmenter
    app.config['predictor'] = predictor
    app.config['device'] = 'cpu'
    app.config['utils'] = utils
    app.config['decoder'] = decoder
    app.config['model'] = model


    db.init_app(app)
    login.init_app(app)
    bootstrap.init_app(app)
    moment.init_app(app)

    from app.errors import bp as errors_bp
    app.register_blueprint(errors_bp)

    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')
    

    from app.classroom import bp as classroom_bp
    app.register_blueprint(classroom_bp, url_prefix='/classroom')
    
    from app.video import bp as video_bp
    app.register_blueprint(video_bp, url_prefix='/video')

    # from app.student import bp as student_bp
    # app.register_blueprint(student_bp, url_prefix='/student')
    

    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    return app

from app import models
