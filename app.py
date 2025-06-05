from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from utils.disease_info import get_disease_info
import sqlite3
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'super_secret_key'

model = load_model("models/lung_disease_densenet.keras")
classes = ['Cancer', 'Covid19', 'Normal', 'Pneumonia', 'Tuberculosis']
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def create_user_table():
    conn = sqlite3.connect('users.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        );
    ''')
    conn.commit()
    conn.close()


def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password)).fetchone()
        conn.close()
        if user:
            session['logged_in'] = True
            session['user_email'] = email
            return redirect(url_for('upload'))
        else:
            flash('Invalid email or password')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, password))
            conn.commit()
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already registered.')
        finally:
            conn.close()
    return render_template('register.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            pred_index = np.argmax(prediction)
            predicted_class = classes[pred_index]
            confidence = float(prediction[0][pred_index])

            report = get_disease_info(predicted_class)
            pdf_filename = generate_pdf_report(filepath, predicted_class, confidence, report)

            return render_template('result.html',
                                   disease=predicted_class,
                                   confidence=confidence,
                                   report=report,
                                   pdf_filename=pdf_filename)

    return render_template('upload.html')


@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


def generate_pdf_report(original_path, predicted_class, confidence, report_info):
    img = cv2.imread(original_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 50, 150)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    gray_path = os.path.join(UPLOAD_FOLDER, 'gray.jpg')
    edge_path = os.path.join(UPLOAD_FOLDER, 'edge.jpg')
    thresh_path = os.path.join(UPLOAD_FOLDER, 'thresh.jpg')
    cv2.imwrite(gray_path, gray)
    cv2.imwrite(edge_path, edge)
    cv2.imwrite(thresh_path, thresh)

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    pdf_filename = f"report_{now}.pdf"
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 20)
    c.drawString(40, height - 50, "Lung Disease Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(40, height - 90, f"Prediction: {predicted_class}")
    c.drawString(40, height - 110, f"Confidence: {confidence:.2f}")
    c.drawString(40, height - 130, f"Future Symptoms: {report_info.get('future_symptoms', '')}")
    c.drawString(40, height - 150, "Suggested Doctors:")
    y = height - 170
    for doc in report_info.get('doctors', []):
        c.drawString(60, y, f"- {doc}")
        y -= 15

    c.drawImage(ImageReader(original_path), 40, height - 380, width=180, height=180)
    c.drawImage(ImageReader(gray_path), 230, height - 380, width=180, height=180)
    c.drawImage(ImageReader(edge_path), 420, height - 380, width=180, height=180)
    c.drawImage(ImageReader(thresh_path), 40, height - 580, width=180, height=180)

    c.save()
    return pdf_filename


if __name__ == '__main__':
    create_user_table()
    app.run(debug=True)
