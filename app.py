from flask import Flask, request, render_template,redirect, send_from_directory, url_for, session, flash
import os
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from datetime import datetime
import tempfile
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# Define the static folder for storing audio files
STATIC_FOLDER = 'static/audio'
os.makedirs(STATIC_FOLDER, exist_ok=True)




# Set up MySQL connection
app.config['MYSQL_HOST'] = 'localhost'  # MySQL server (usually localhost)
app.config['MYSQL_USER'] = 'root'  # Your MySQL username
app.config['MYSQL_PASSWORD'] = ''  # Your MySQL password
app.config['MYSQL_DB'] = 'audio_app'  # Your database name
app.config['SECRET_KEY'] = os.urandom(24)  # Secret key for session

mysql = MySQL(app)

# Route for home page (redirect to login page if not logged in)
@app.route('/')
def index():
    if 'logged_in' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('register'))

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()

        if user and check_password_hash(user[6], password):  # user[6] is the password hash
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    
    return render_template('login.html')

# Route for registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        phone_number = request.form['phone_number']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match. Please try again.', 'danger')
            return redirect(url_for('register'))
        
        # Hash the password
        hashed_password = generate_password_hash(password)

        # Check if the username or email already exists
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE username = %s OR email = %s', (username, email))
        existing_user = cursor.fetchone()
        
        if existing_user:
            flash('Username or email already exists. Try another one.', 'danger')
        else:
            # Insert the new user into the database
            cursor.execute('INSERT INTO users (username, first_name, last_name, email, phone_number, password, confirm_password) '
                           'VALUES (%s, %s, %s, %s, %s, %s, %s)', 
                           (username, first_name, last_name, email, phone_number, hashed_password, hashed_password))
            mysql.connection.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')




# Function to load audio file
def load_audio(file_path):
    """Load an audio file using librosa."""
    audio, sr = librosa.load(file_path, sr=None)  # sr=None to retain original sample rate
    return audio, sr

# Noise reduction function using noisereduce
def noise_reduction(audio, sr):
    """Reduce noise using the noisereduce library (spectral gating)."""
    denoised_audio = nr.reduce_noise(y=audio, sr=sr)
    return denoised_audio

# Function to save the denoised audio
def save_audio(file_path, audio, sr):
    """Save the denoised audio using soundfile."""
    sf.write(file_path, audio, sr)
    print(f"Audio saved to {file_path}")

# Function to create a unique timestamped file name
def generate_output_file_path(output_folder, filename):
    """Generate a unique output file path with a timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_folder, f"{filename}_{timestamp}.wav")


# Route to handle file upload, denoising, and playing audio
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return "No file part"
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return "No selected file"
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        audio_file.save(temp_file.name)
        input_file_path = temp_file.name
        
        # Load the noisy audio
        audio, sr = load_audio(input_file_path)
        
        # Perform noise reduction
        denoised_audio = noise_reduction(audio, sr)
        
        # Generate output paths for both original and denoised audio
        original_audio_file = os.path.join(STATIC_FOLDER, 'original_audio.wav')
        denoised_audio_file = generate_output_file_path(STATIC_FOLDER, 'denoised_audio')
        
        # Save the original and denoised audio files in static folder
        save_audio(original_audio_file, audio, sr)
        save_audio(denoised_audio_file, denoised_audio, sr)
        
        # Generate URLs for the original and denoised audio files
        original_audio_url = url_for('static', filename='audio/original_audio.wav')
        denoised_audio_url = url_for('static', filename=f'audio/{os.path.basename(denoised_audio_file)}')

        # Return the HTML page with audio players
        return render_template('play_audio.html', original_audio_url=original_audio_url, denoised_audio_url=denoised_audio_url)

# Route to serve static files (original and denoised audio)
@app.route('/static/audio/<filename>')
def serve_static_audio(filename):
    return send_from_directory(STATIC_FOLDER, filename)

# Route to log out
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
