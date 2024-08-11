# app.py
import os
from dotenv import load_dotenv
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, current_app
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from deepface import DeepFace
import numpy as np

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'fallback-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mood_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = str(Path.home() / 'mood_app_uploads')

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def analyze_text_mood(text):
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def analyze_facial_emotion(image_path):
    try:
        result = DeepFace.analyze(image_path, actions=['emotion'])
        return result[0]['dominant_emotion']
    except Exception as e:
        current_app.logger.error(f"Error in facial analysis: {str(e)}")
        return None

def get_recommendation(mood):
    recommendations = {
        'positive': [
            "Great mood! Why not try a new hobby or learn something exciting?",
            "Your positive energy is contagious! Spread joy by doing something kind for others.",
            "Perfect time for some upbeat music and dance! It'll boost your mood even more."
        ],
        'negative': [
            "Take a moment for yourself. Deep breathing or meditation might help.",
            "Reach out to a friend or loved one. Talking can often lift our spirits.",
            "Try some light exercise or a short walk. Physical activity can help improve mood."
        ],
        'neutral': [
            "How about exploring a new podcast or audiobook?",
            "This could be a good time for some creative activities like drawing or writing.",
            "Consider planning a future activity or trip. It can give you something to look forward to!"
        ],
        'happy': [
            "Your happiness is wonderful! Why not share it by doing something nice for someone?",
            "Capture this feeling! Journal about what's making you happy right now.",
            "Use this positive energy for something productive you've been wanting to do."
        ],
        'sad': [
            "It's okay to feel sad. Be kind to yourself and do something soothing.",
            "Listen to music that resonates with you right now. Music can be very therapeutic.",
            "Consider talking to a friend or professional if the sadness persists."
        ],
        'angry': [
            "Take a pause and practice some deep breathing exercises.",
            "Physical activity can help release tension. Maybe try a quick workout?",
            "Write down your feelings. It can help process emotions and find solutions."
        ],
        'surprise': [
            "Unexpected things can be exciting! Why not surprise someone else positively today?",
            "Channel this energy into trying something new or spontaneous!",
            "Reflect on what surprised you and what you can learn from it."
        ],
        'fear': [
            "Remember that it's okay to feel this way. Try some grounding exercises.",
            "Reach out to someone you trust. Sharing your fears can often help.",
            "Consider writing down your worries and then challenging them rationally."
        ],
        'disgust': [
            "Focus on something pleasant. Maybe look at some beautiful nature photos.",
            "If something's bothering you, consider if there's a constructive way to address it.",
            "Engage in an activity you enjoy to shift your focus and mood."
        ]
    }
    return np.random.choice(recommendations.get(mood, recommendations['neutral']))

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            login_user(user)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registered successfully. Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    text = request.form.get('text', '')
    image = request.files.get('image')

    text_mood = analyze_text_mood(text) if text else None
    
    facial_emotion = None
    if image and allowed_file(image.filename):
        try:
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            
            if os.path.exists(image_path):
                facial_emotion = analyze_facial_emotion(image_path)
                os.remove(image_path)  # Remove the image after analysis
            else:
                current_app.logger.error(f"Failed to save image: {image_path}")
        except Exception as e:
            current_app.logger.error(f"Error processing image: {str(e)}")
    elif image:
        current_app.logger.warning(f"Invalid file type uploaded: {image.filename}")

    # Combine text and image analysis
    final_mood = facial_emotion if facial_emotion else (text_mood if text_mood else 'neutral')
    recommendation = get_recommendation(final_mood)

    return jsonify({
        'text_mood': text_mood,
        'facial_emotion': facial_emotion,
        'final_mood': final_mood,
        'recommendation': recommendation
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        print(f"Upload directory: {upload_dir}")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Test write permissions
        test_file = upload_dir / 'test_write.txt'
        try:
            test_file.write_text('Test write permissions')
            print("Successfully wrote to upload directory")
            test_file.unlink()  # Remove the test file
        except PermissionError:
            print("ERROR: No write permission in upload directory")
        except Exception as e:
            print(f"ERROR: Unexpected issue with upload directory: {str(e)}")
    
    app.run(debug=True)
