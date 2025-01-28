from flask import Flask, request, jsonify, render_template, session
from werkzeug.security import generate_password_hash, check_password_hash
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import cv2
import pytesseract
import pandas as pd
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
users = {}
def extract_metadata(image_path):
    """Extract metadata from ECG image using OCR."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    print(f"Extracted Text from {image_path}:\n{text}")

    metadata = {
        "age": 0,
        "gender": "Unknown",
        "height": 0,
        "weight": 0,
    }

    if "Years" in text:
        age_line = [line for line in text.split("\n") if "Years" in line]
        if age_line:
            metadata["age"] = int(age_line[0].split()[1])
    if "Male" in text or "Female" in text:
        metadata["gender"] = "Male" if "Male" in text else "Female"

    return metadata

def preprocess_image(image_path):
    """Preprocess ECG image for input into CNN."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0  # Normalize
    return image

def load_dataset(folder_path):
    """Load ECG images and metadata."""
    images = []
    metadata = []
    labels = []

    for file_path in os.listdir(folder_path):
        if file_path.endswith('.jpg'):
            image_path = os.path.join(folder_path, file_path)
            image = preprocess_image(image_path)
            meta = extract_metadata(image_path)
            label = np.random.choice(["Normal", "Abnormal"])

            images.append(image)
            metadata.append(meta)
            labels.append(label)

    return np.array(images), pd.DataFrame(metadata), labels

def train_model(images, metadata, labels):
    """Train a combined CNN + metadata model."""
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    categorical_labels = to_categorical(encoded_labels)

    X_train_img, X_test_img, X_train_meta, X_test_meta, y_train, y_test = train_test_split(
        images, metadata, categorical_labels, test_size=0.2, random_state=42
    )

    image_model = Sequential([
        Input(shape=(256, 256, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
    ])

    metadata_input_shape = X_train_meta.shape[1]
    metadata_model = Sequential([
        Input(shape=(metadata_input_shape,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
    ])

    combined = tf.keras.layers.concatenate([image_model.output, metadata_model.input])
    final_output = Dense(64, activation='relu')(combined)
    final_output = Dense(categorical_labels.shape[1], activation='softmax')(final_output)

    model = tf.keras.Model(inputs=[image_model.input, metadata_model.input], outputs=final_output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        [X_train_img, X_train_meta],
        y_train,
        epochs=20,
        validation_split=0.2,
        batch_size=32
    )

    test_loss, test_acc = model.evaluate([X_test_img, X_test_meta], y_test)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    return model, label_encoder
ecg_model = None
label_encoder = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.form
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'success': False, 'message': 'Please fill in all fields'})

    if email in users:
        return jsonify({'success': False, 'message': 'Email already exists'})

    hashed_password = generate_password_hash(password)
    users[email] = {'username': username, 'password': hashed_password}

    return jsonify({'success': True, 'message': 'User registered successfully'})

@app.route('/api/signin', methods=['POST'])
def api_signin():
    data = request.form
    email = data.get('email')
    password = data.get('password')

    user = users.get(email)
    if user and check_password_hash(user['password'], password):
        session['user_email'] = email
        session['username'] = user['username']
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': 'Invalid email or password'})

@app.route('/api/signout')
def api_signout():
    session.clear()
    return jsonify({'success': True})

@app.route('/api/check_auth')
def api_check_auth():
    if 'user_email' in session:
        return jsonify({'authenticated': True})
    else:
        return jsonify({'authenticated': False})

@app.route('/api/process_ecg', methods=['POST'])
def api_process_ecg():
    global ecg_model, label_encoder

    if 'user_email' not in session:
        return jsonify({'success': False, 'message': 'User not authenticated'})

    if 'ecgFile' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})

    file = request.files['ecgFile']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})

    if file:
        temp_path = 'temp_ecg.jpg'
        file.save(temp_path)
        image = preprocess_image(temp_path)
        image = np.expand_dims(image, axis=0)  
        metadata = extract_metadata(temp_path)
        metadata_array = np.array([[metadata['age'], metadata['gender'] == 'Male', metadata['height'], metadata['weight']]])
        if ecg_model is None or label_encoder is None:
            folder_path = 'path_to_ecg_folder' 
            images, metadata_df, labels = load_dataset(folder_path)
            ecg_model, label_encoder = train_model(images, metadata_df, labels)
        prediction = ecg_model.predict([image, metadata_array])
        predicted_class = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        heart_rate = np.random.randint(60, 100) 
        risk_level = "Low" if predicted_label == "Normal" else "High"
        doctor_consult = "Yes" if predicted_label == "Abnormal" else "No"
        recommendations = {
            'exercise': [
                'Include light cardio exercises 3-4 times a week',
                'Practice yoga or tai chi for stress reduction and flexibility'
            ],
            'diet': [
                'Include more fruits and vegetables in your diet',
                'Reduce sodium consumption'
            ],
            'lifestyle': [
                'Aim for 7-8 hours of sleep per night',
                'Practice stress-reduction techniques like meditation'
            ]
        }
        os.remove(temp_path)
        result = {
            'heartRate': int(heart_rate),
            'heartBeat': predicted_label,
            'doctorConsult': doctor_consult,
            'riskLevel': risk_level,
            'stages': {
                'Normal': float(prediction[0][0]),
                'Abnormal': float(prediction[0][1])
            },
            'recommendations': recommendations
        }

        return jsonify({'success': True, 'message': 'ECG analysis completed successfully', 'data': result})

if __name__ == '__main__':
    app.run(debug=True)

