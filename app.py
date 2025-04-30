from flask import Flask, render_template, request, redirect, url_for, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from datetime import datetime
import os
import gdown
from fpdf import FPDF  # Using fpdf2 only

# Google Drive model setup
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "kidney_model.h5")
DRIVE_FILE_ID = "1oyb1r2OXFXSbxX2YPhiZ534OMbogpcCI"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# Load model
model = load_model(MODEL_PATH)
labels = ['Cyst', 'Normal', 'Stone', 'Tumor']

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database and login manager
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Prediction record model
class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(200), nullable=False)

# Load user for login session
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        if User.query.filter_by(username=username).first():
            return 'Username already exists!'
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            return 'Invalid username or password!'
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file!', 400

    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)[0]
    predicted_class = labels[np.argmax(preds)]
    confidence = round(np.max(preds) * 100, 2)

    # Save prediction to database
    history = PredictionHistory(
        user_id=current_user.id,
        prediction=predicted_class,
        confidence=confidence,
        image_path=os.path.join('uploads', filename)
    )
    db.session.add(history)
    db.session.commit()

    class_confidences = {
        label: round(score * 100, 2) for label, score in zip(labels, preds)
    }

    return render_template(
        'result.html',
        prediction=predicted_class,
        confidence=confidence,
        image_path=url_for('static', filename='uploads/' + filename),
        class_confidences=class_confidences,
        record_id=history.id
    )

@app.route('/history')
@login_required
def history():
    records = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.date.desc()).all()
    return render_template('history.html', records=records)

@app.route('/download_report/<int:record_id>')
@login_required
def download_report(record_id):
    record = PredictionHistory.query.get_or_404(record_id)
    if record.user_id != current_user.id:
        return 'Unauthorized', 403

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Kidney Disease Prediction Report', ln=True, align='C')
    pdf.set_font('Arial', '', 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Prediction: {record.prediction}", ln=True)
    pdf.cell(0, 10, f"Confidence: {record.confidence}%", ln=True)
    pdf.cell(0, 10, f"Date: {record.date.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Image Path: {record.image_path}", ln=True)

    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    return send_file(pdf_output, as_attachment=True, download_name='report.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
