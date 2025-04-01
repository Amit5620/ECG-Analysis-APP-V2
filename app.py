from flask import Flask, render_template, request, redirect, url_for, session
import os
import cv2
import numpy as np
from roboflow import Roboflow
import supervision as sv
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['MYSQL_HOST'] = 'sql12.freesqldatabase.com'
app.config['MYSQL_USER'] = 'sql12770762'
app.config['MYSQL_PASSWORD'] = 'VggMxN1TtD'
app.config['MYSQL_DB'] = 'sql12770762'

mysql = MySQL(app)


# Fix: Set upload folder in app.config
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# Initialize Roboflow Models
rf = Roboflow(api_key="kqlgTsdyBapHPYnoxznG")
project1 = rf.workspace().project("ecg-classification-ygs4v")
model1 = project1.version(1).model

project2 = rf.workspace().project("ecg_detection")
model2 = project2.version(3).model

# Home Route
@app.route('/')
def home():
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    else:
        return render_template('login.html')

# Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute(f"insert into users (username, email, password) values('{username}', '{email}', '{password}')")
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('login'))
    return render_template('register.html')


# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        pwd = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute(f"select username, password from users where username = '{username}'")
        user = cur.fetchone()
        cur.close()
        if user and pwd == user[1]:
            session['username'] = user[0]
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid Credentials")
    
    return render_template('login.html')
        


# Dashboard Route
@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('index.html')
    return redirect(url_for('home'))


# Upload Page Route
@app.route('/upload')
def upload():
    if 'username' in session:
        return render_template('upload.html')
    return redirect(url_for('home'))


# Prediction Route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Validate File Type (only allow images)
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return "Invalid file type", 400

    # Secure the filename and avoid overwriting
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)

    # Resize the image
    image = cv2.imread(file_path)
    image = cv2.resize(image, (500, 500))
    cv2.imwrite(file_path, image)

    # Retrieve patient details safely
    patient_name = request.form.get('patient_name', '').strip()
    patient_age = request.form.get('patient_age', '').strip()
    patient_gender = request.form.get('patient_gender', '').strip()
    patient_id = request.form.get('patient_id', '').strip()
    doctor_name = request.form.get('doctor_name', '').strip()
    symptoms = request.form.get('patient_symptoms', '').strip()

    # Check if required fields are missing
    if not all([patient_name, patient_age, patient_gender, patient_id, doctor_name, symptoms]):
        return "Missing patient details", 400

    # Run model predictions
    result1 = model1.predict(file_path, confidence=40, overlap=30).json()
    result2 = model2.predict(file_path, confidence=40, overlap=30).json()

    def process_result(result, file_path, suffix):
        predictions = result.get("predictions", [])
        if not predictions:
            return unique_filename, "No abnormality detected"

        labels = [pred["class"] for pred in predictions]
        output_filename = f"{os.path.splitext(unique_filename)[0]}{suffix}.jpg"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        # Draw bounding boxes
        image = cv2.imread(file_path)
        for pred in predictions:
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            cv2.rectangle(image, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
        cv2.imwrite(output_path, image)

        return output_filename, labels

    # Process predictions
    pred_img1, predicted_classes1 = process_result(result1, file_path, '-pred1')
    pred_img2, predicted_classes2 = process_result(result2, file_path, '-pred2')

    # Store in MySQL Database
    cur = mysql.connection.cursor()

    # Prepare the SQL query with placeholders
    query = """
        INSERT INTO predictions (username, patient_name, patient_age, patient_gender, patient_id, doctor_name, symptoms, uploaded_image, prediction_result, prediction_image1, prediction_image2) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    # Prepare the data to be inserted
    data = (
        session['username'],
        patient_name,
        patient_age,
        patient_gender,
        patient_id,
        doctor_name,
        symptoms,
        unique_filename,
        predicted_classes1,
        pred_img1,
        pred_img2
    )

    # Execute the query with the data
    cur.execute(query, data)

    # Commit the transaction
    mysql.connection.commit()

    # Close the cursor
    cur.close()

    return render_template('result.html', 
                           original=unique_filename, pred1=pred_img1, pred2=pred_img2,
                           patient_name=patient_name, patient_age=patient_age, 
                           patient_gender=patient_gender, patient_id=patient_id, doctor_name=doctor_name, patient_symptoms=symptoms,
                           predicted_classes1=predicted_classes1, predicted_classes2=predicted_classes2)


# Logout Route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
