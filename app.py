import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO
from werkzeug.security import generate_password_hash, check_password_hash

# --- App & DB Initialization ---
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# --- Database Models (User, Service Center, Booking) ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone_number = db.Column(db.String(20), nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    def set_password(self, p): self.password_hash = generate_password_hash(p)
    def check_password(self, p): return check_password_hash(self.password_hash, p)

class ServiceCenter(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    center_name = db.Column(db.String(150), unique=True, nullable=False)
    mobile_number = db.Column(db.String(20), nullable=False)
    address = db.Column(db.String(250))
    contact_email = db.Column(db.String(120))
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    slots = db.relationship('TimeSlot', backref='service_center', lazy=True, cascade="all, delete-orphan")
    bookings = db.relationship('Booking', backref='service_center', lazy=True, cascade="all, delete-orphan")

class TimeSlot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    slot_time = db.Column(db.String(50), nullable=False)
    is_booked = db.Column(db.Boolean, default=False)
    service_center_id = db.Column(db.Integer, db.ForeignKey('service_center.id'), nullable=False)
    booking = db.relationship('Booking', backref='timeslot', uselist=False, cascade="all, delete-orphan")

class Booking(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user_name = db.Column(db.String(100), nullable=False)
    time_slot_id = db.Column(db.Integer, db.ForeignKey('time_slot.id'), unique=True, nullable=False)
    service_center_id = db.Column(db.Integer, db.ForeignKey('service_center.id'), nullable=False)
    task = db.Column(db.String(100), nullable=False)
    booking_time = db.Column(db.DateTime, default=datetime.utcnow)

def create_default_slots(center):
    time_slots = ["09:00 AM", "10:00 AM", "11:00 AM", "12:00 PM", "02:00 PM", "03:00 PM", "04:00 PM", "05:00 PM"]
    for time in time_slots:
        slot = TimeSlot(slot_time=time, service_center_id=center.id)
        db.session.add(slot)
    db.session.commit()


# --- MERGED: YOLO Model and Cost Estimation Logic ---
MODEL_PATH = 'best.pt'
model = YOLO(MODEL_PATH)

# Load cost data from CSV
df = pd.read_csv('deepseek_csv_20251011_c1afe8.txt')

# Damage class to part mapping
DAMAGE_TO_PART = {
    'lamp-crack': ['Head Light', 'Tail Light', 'Fog Lamp'],
    'glass-crack': ['Front Windshield Glass', 'Rear Windshield Glass'],
    'side-mirror-crack': ['Side View Mirror', 'Rear View Mirror'],
    'flat-tire': ['Alloy Wheel Front', 'Alloy Wheel Rear'],
    'car-part-crack': ['Bonnet/Hood', 'Front Bumper', 'Rear Bumper', 'Fender'],
    'scratches': ['Front Door', 'Rear Door', 'Bonnet/Hood'],
    'paint-chips': ['Front Door', 'Rear Door', 'Fender'],
    'minor-deformation': ['Front Bumper', 'Rear Bumper'],
    'moderate-deformation': ['Front Bumper', 'Rear Bumper', 'Bonnet/Hood'],
    'severe-deformation': ['Front Bumper', 'Rear Bumper', 'Bonnet/Hood', 'Fender'],
    'detachment': ['Front Bumper', 'Rear Bumper']
}

# Severity thresholds
SEVERITY_THRESHOLDS = {
    'scratches': 0.2,
    'paint-chips': 0.3,
    'minor-deformation': 0.5,
    'flat-tire': 0.6,
    'lamp-crack': 0.6,
    'car-part-crack': 0.7,
    'side-mirror-crack': 0.7,
    'moderate-deformation': 0.7,
    'glass-crack': 0.9,
    'detachment': 0.9,
    'severe-deformation': 1.0
}

def calculate_severity(damage_class, confidence):
    base_severity = SEVERITY_THRESHOLDS.get(damage_class, 0.5)
    severity_percentage = (base_severity * 0.7 + confidence * 0.3) * 100
    return min(severity_percentage, 100)

def decide_action(severity_percentage):
    if severity_percentage >= 70:
        return "REPLACE"
    else:
        return "REPAIR"

def get_cost_estimate(car_brand, car_model, year_range, damage_class, action):
    car_data = df[
        (df['car_brand'].str.lower() == car_brand.lower()) &
        (df['car_model'].str.lower() == car_model.lower())
    ]
    if car_data.empty:
        return [] # Return empty list instead of None
    
    possible_parts = DAMAGE_TO_PART.get(damage_class, [])
    cost_estimates = []
    for part in possible_parts:
        part_data = car_data[car_data['part_name'].str.contains(part, case=False, na=False)]
        if not part_data.empty:
            row = part_data.iloc[0]
            if action == "REPAIR":
                cost_estimates.append({
                    'part_name': row['part_name'], 'part_price': float(row['part_price']),
                    'labour_cost': float(row['repair_labour_cost']), 'total_cost': float(row['total_repair_cost']),
                    'category': row['part_category']
                })
            else:
                cost_estimates.append({
                    'part_name': row['part_name'], 'part_price': float(row['part_price']),
                    'labour_cost': float(row['replace_labour_cost']), 'total_cost': float(row['total_replace_cost']),
                    'category': row['part_category']
                })
    return cost_estimates


# --- API Endpoints ---

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    role = data.get('role')
    if role == 'user':
        email = data.get('email')
        if not all([data.get('name'), email, data.get('phone_number'), data.get('password')]):
            return jsonify({'error': 'All user fields are required'}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'User with this email already exists'}), 409
        new_user = User(name=data.get('name'), email=email, phone_number=data.get('phone_number'))
        new_user.set_password(data.get('password'))
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    elif role == 'service_center':
        center_name = data.get('center_name')
        if not all([center_name, data.get('mobile_number'), data.get('latitude'), data.get('longitude')]):
            return jsonify({'error': 'Required fields are missing'}), 400
        if ServiceCenter.query.filter_by(center_name=center_name).first():
            return jsonify({'error': 'A service center with this name already exists'}), 409
        new_center = ServiceCenter(
            center_name=center_name, mobile_number=data.get('mobile_number'),
            address=data.get('address'), contact_email=data.get('contact_email'),
            latitude=data.get('latitude'), longitude=data.get('longitude')
        )
        db.session.add(new_center)
        db.session.commit() 
        create_default_slots(new_center)
        return jsonify({'message': 'Service center registered successfully'}), 201
    return jsonify({'error': 'Invalid role'}), 400

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    role = data.get('role')
    if role == 'user':
        user = User.query.filter_by(email=data.get('email')).first()
        if user and user.check_password(data.get('password')):
            return jsonify({'message': 'Login successful', 'role': 'user', 'user_id': user.id, 'name': user.name}), 200
    elif role == 'service_center':
        center = ServiceCenter.query.filter_by(center_name=data.get('center_name'), mobile_number=data.get('mobile_number')).first()
        if center:
            return jsonify({'message': 'Login successful', 'role': 'service_center', 'center_id': center.id, 'name': center.center_name}), 200
    return jsonify({'error': 'Invalid credentials'}), 401

# MERGED: The complete /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files: return jsonify({'error': 'No image provided'}), 400
        
        # Get form data
        car_brand = request.form.get('car_brand')
        car_model = request.form.get('car_model')
        year = request.form.get('year')
        
        # Read image
        image_file = request.files['image']
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run YOLO prediction
        results = model.predict(source=img, imgsz=640, conf=0.25, save=False)
        
        if not results or len(results[0].boxes) == 0:
            return jsonify({'message': 'No damage detected'}), 200
        
        # Get highest confidence detection
        res = results[0]
        max_idx = np.argmax(res.boxes.conf.cpu().numpy())
        highest_conf_box = res.boxes[max_idx]
        
        damage_class = model.names[int(highest_conf_box.cls)]
        confidence = float(highest_conf_box.conf)
        
        # Calculate severity, action, and cost
        severity_percentage = calculate_severity(damage_class, confidence)
        action = decide_action(severity_percentage)
        cost_estimates = get_cost_estimate(car_brand, car_model, year, damage_class, action)
        
        response = {
            'damage_class': damage_class,
            'confidence': round(confidence * 100, 2),
            'severity_percentage': round(severity_percentage, 2),
            'recommended_action': action,
            'cost_estimates': cost_estimates,
            'car_info': {'brand': car_brand, 'model': car_model, 'year': year}
        }
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error in /predict: {e}") # Log the error for debugging
        return jsonify({'error': str(e)}), 500

@app.route('/service-centers', methods=['GET'])
def get_service_centers():
    centers = ServiceCenter.query.all()
    return jsonify([{"id": c.id, "center_name": c.center_name, "mobile_number": c.mobile_number, "address": c.address, "contact_email": c.contact_email, "latitude": c.latitude, "longitude": c.longitude} for c in centers]), 200

@app.route('/service-centers/<int:center_id>/slots', methods=['GET'])
def get_slots(center_id):
    slots = TimeSlot.query.filter_by(service_center_id=center_id, is_booked=False).order_by(TimeSlot.id).all()
    return jsonify([{"id": s.id, "time": s.slot_time, "is_booked": s.is_booked} for s in slots]), 200

@app.route('/service-centers/<int:center_id>/bookings', methods=['GET'])
def get_bookings(center_id):
    bookings = db.session.query(Booking, TimeSlot).join(TimeSlot).filter(Booking.service_center_id == center_id).all()
    return jsonify([{"booking_id": b.id, "user_name": b.user_name, "slot_time": s.slot_time, "task": b.task} for b, s in bookings]), 200

@app.route('/bookings', methods=['POST'])
def create_booking():
    data = request.json
    user = User.query.get(data.get('user_id'))
    if not user: return jsonify({'error': 'User not found'}), 404

    slot = TimeSlot.query.get(data.get('slot_id'))
    if not slot or slot.is_booked: return jsonify({'error': 'Slot is not available'}), 400
    
    slot.is_booked = True
    new_booking = Booking(
        user_id=user.id, user_name=user.name, time_slot_id=slot.id, 
        service_center_id=slot.service_center_id, task=data.get('task', 'General Checkup')
    )
    db.session.add(new_booking)
    db.session.commit()
    return jsonify({'message': 'Booking successful'}), 201

# --- Main Execution ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5001, debug=True)