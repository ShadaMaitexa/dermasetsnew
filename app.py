from flask import Flask, request, jsonify, render_template, redirect
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import logging
import cv2
import numpy as np
import gc

# Memory optimization: limit torch threads
torch.set_num_threads(1)

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False


from flask_cors import CORS

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app) # Enable CORS for Flutter integration

# Configure logging at top level for Gunicorn
logging.basicConfig(level=logging.INFO)


# ================================
# Firebase Initialization
# ================================
db = None
if FIREBASE_AVAILABLE:
    cred_path = os.path.join(BASE_DIR, 'serviceAccountKey.json')

    if os.path.exists(cred_path):
        try:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("Firebase Admin initialized successfully.")
        except Exception as e:
            print(f"Firebase initialization error: {e}")
    else:
        print("serviceAccountKey.json not found. Firebase disabled.")


# ================================
# Model Configuration
# ================================
num_classes = 5
img_width, img_height = 150, 150
CONFIDENCE_THRESHOLD = 75


transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ================================
# Load Model
# ================================
# Using weights=None instead of deprecated pretrained=False
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

model_path = os.path.join(BASE_DIR, 'models', 'skin_disease_model.pth')

if os.path.exists(model_path):
    try:
        # Load model to CPU explicitly
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        del state_dict # Free memory
        gc.collect()   # Force garbage collection
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model load error: {e}")
else:
    print("Model file not found. Using untrained model.")

model.eval()


# ================================
# Class Labels
# ================================
class_labels = {
    0: 'Acne',
    1: 'Hairloss',
    2: 'Nail Fungus',
    3: 'Normal',
    4: 'Skin Allergy'
}


# ================================
# Disease Details
# ================================
DISEASE_DETAILS = {

    'Acne': {
        'description':
            'Acne is caused by clogged pores and oil buildup.',

        'steps': [
            {
                'title': 'Gentle Cleansing',
                'description': 'Wash face twice daily.',
                'icon': 'wash'
            },
            {
                'title': 'Topical Treatment',
                'description': 'Use salicylic acid products.',
                'icon': 'science'
            },
            {
                'title': 'Avoid Touching',
                'description': 'Do not pick pimples.',
                'icon': 'do_not_disturb_on'
            }
        ]
    },

    'Hairloss': {
        'description': 'Hair loss due to genetics or nutrition.',

        'steps': [
            {
                'title': 'Nutrition',
                'description': 'Increase iron and biotin.',
                'icon': 'restaurant'
            },
            {
                'title': 'Massage',
                'description': 'Improve blood circulation.',
                'icon': 'touch_app'
            },
            {
                'title': 'Consult Doctor',
                'description': 'Visit dermatologist.',
                'icon': 'medical_services'
            }
        ]
    },

    'Nail Fungus': {
        'description': 'Fungal infection affecting nails.',

        'steps': [
            {
                'title': 'Keep Dry',
                'description': 'Avoid moisture.',
                'icon': 'opacity'
            },
            {
                'title': 'Antifungal Cream',
                'description': 'Apply medication.',
                'icon': 'medical_information'
            },
            {
                'title': 'Trim Nails',
                'description': 'Keep nails short.',
                'icon': 'content_cut'
            }
        ]
    },

    'Skin Allergy': {
        'description': 'Skin reaction to allergens.',

        'steps': [
            {
                'title': 'Identify Trigger',
                'description': 'Find allergen.',
                'icon': 'search'
            },
            {
                'title': 'Cool Compress',
                'description': 'Reduce irritation.',
                'icon': 'ac_unit'
            },
            {
                'title': 'Medication',
                'description': 'Use antihistamines.',
                'icon': 'medication'
            }
        ]
    },

    'Normal': {
        'description': 'Skin appears healthy.',

        'steps': [
            {
                'title': 'Hydrate',
                'description': 'Drink water.',
                'icon': 'water_drop'
            },
            {
                'title': 'Sun Protection',
                'description': 'Use sunscreen.',
                'icon': 'wb_sunny'
            },
            {
                'title': 'Maintain Routine',
                'description': 'Continue skincare.',
                'icon': 'shield'
            }
        ]
    }

}


# ================================
# Skin Detection Function
# ================================
def is_skin_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 48, 80], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    skin_pixels = cv2.countNonZero(mask)
    total_pixels = img.shape[0] * img.shape[1]

    ratio = skin_pixels / total_pixels

    return ratio > 0.15


# ================================
# Prediction Function
# ================================
def predict_skin_disease(image_path):

    # Check if skin exists
    if not is_skin_image(image_path):
        return "Invalid Image (Not Skin)", 0

    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)

    with torch.no_grad():

        outputs = model(img)

        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    confidence_value = confidence.item() * 100

    if confidence_value < CONFIDENCE_THRESHOLD:
        return "Uncertain Prediction", confidence_value

    label = class_labels.get(predicted.item(), "Unknown")

    return label, confidence_value


# ================================
# API Endpoint
# ================================
@app.route('/predict', methods=['POST'])
def predict_api():
    logging.info("Prediction request received")
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    user_id = request.form.get('uid')
    image_url = request.form.get('image_url')

    uploads_dir = os.path.join(BASE_DIR, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    file_path = os.path.join(uploads_dir, file.filename)

    file.save(file_path)

    try:

        label, confidence = predict_skin_disease(file_path)

        details = DISEASE_DETAILS.get(label, {})

        # Save history
        if db and user_id and label not in [
            "Invalid Image (Not Skin)",
            "Uncertain Prediction"
        ]:

            db.collection('user')\
              .document(user_id)\
              .collection('scan_history')\
              .add({

                  'disease_name': label,
                  'confidence': int(confidence),
                  'image_url': image_url or '',
                  'timestamp': firestore.SERVER_TIMESTAMP
              })

        return jsonify({

            'prediction': label,
            'confidence': confidence,
            'details': details,
            'status': 'success'

        })

    except Exception as e:

        logging.error(str(e))

        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200


# ================================
# HTML Upload Endpoint
# ================================
@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':

        file = request.files['file']

        uploads_dir = os.path.join(BASE_DIR, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)

        file_path = os.path.join(uploads_dir, file.filename)

        file.save(file_path)

        label, confidence = predict_skin_disease(file_path)

        return render_template(
            'result.html',
            prediction=label,
            confidence=f"{confidence:.2f}%"
        )

    return render_template('upload.html')


# ================================
# Run App
# ================================
if __name__ == '__main__':

    uploads_dir = os.path.join(BASE_DIR, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
