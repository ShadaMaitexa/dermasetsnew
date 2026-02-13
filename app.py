from flask import Flask, request, jsonify, render_template, redirect
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import logging
from datetime import datetime

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# Base directory for reliable relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# --- Firebase Initialization ---
db = None
if FIREBASE_AVAILABLE:
    # Look for serviceAccountKey.json in the same directory as app.py
    cred_path = os.path.join(BASE_DIR, 'serviceAccountKey.json')
    if os.path.exists(cred_path):
        try:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("Firebase Admin initialized successfully.")
        except Exception as e:
            print(f"Error initializing Firebase Admin: {e}")
    else:
        print("Warning: serviceAccountKey.json not found. Firebase history recording disabled.")

# --- Model Loading ---
num_classes = 5
img_width, img_height = 150, 150

# Defining transform
transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize model structure
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Try to load model weights
model_path = os.path.join(BASE_DIR, 'models', 'skin_disease_model.pth')
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
else:
    print(f"Warning: Model file not found at {model_path}. Predictions will be random/untrained.")
    model.eval()

# Disease Information Map
DISEASE_DETAILS = {
    'Acne': {
        'description': 'Acne is a common skin condition where pores become clogged with oil and dead skin cells, leading to pimples, blackheads, or cysts.',
        'steps': [
            {'title': 'Gentle Cleansing', 'description': 'Wash your face twice daily with a mild cleanser.', 'icon': 'wash'},
            {'title': 'Topical Treatment', 'description': 'Use products containing salicylic acid or benzoyl peroxide.', 'icon': 'science'},
            {'title': 'Avoid Touching', 'description': 'Do not squeeze or pick at pimples to prevent scarring.', 'icon': 'do_not_disturb_on'}
        ]
    },
    'Hairloss': {
        'description': 'Hair loss can be caused by genetics, hormonal changes, medical conditions, or aging. It can affect just your scalp or your entire body.',
        'steps': [
            {'title': 'Check Nutrition', 'description': 'Ensure adequate intake of iron, zinc, and biotin.', 'icon': 'restaurant'},
            {'title': 'Scalp Massage', 'description': 'Can help improve blood circulation to hair follicles.', 'icon': 'touch_app'},
            {'title': 'Consult Specialist', 'description': 'See a dermatologist to identify the specific type of hair loss.', 'icon': 'medical_services'}
        ]
    },
    'Nail Fungus': {
        'description': 'A common condition that begins as a white or yellow spot under the tip of your fingernail or toenail.',
        'steps': [
            {'title': 'Keep Dry', 'description': 'Keep your hands and feet dry and clean.', 'icon': 'opacity'},
            {'title': 'Antifungal Cream', 'description': 'Apply over-the-counter antifungal nail creams.', 'icon': 'medical_information'},
            {'title': 'Trim Nails', 'description': 'Keep nails short and thin to reduce pressure.', 'icon': 'content_cut'}
        ]
    },
    'Skin Allergy': {
        'description': 'A skin allergy occurs when your skin comes into contact with an allergen, causing an itchy, red rash or hives.',
        'steps': [
            {'title': 'Identify Triggers', 'description': 'Track what you touched or ate before the reaction.', 'icon': 'search'},
            {'title': 'Cool Compress', 'description': 'Apply a cool, damp cloth to soothe the itching.', 'icon': 'ac_unit'},
            {'title': 'Antihistamines', 'description': 'Consider OTC antihistamines for relief.', 'icon': 'medication'}
        ]
    },
    'Normal': {
        'description': 'Your skin appears healthy with good balance and no significant signs of disease in the analyzed area.',
        'steps': [
            {'title': 'Hydration', 'description': 'Continue using a good moisturizer and drinking water.', 'icon': 'water_drop'},
            {'title': 'Sun Protection', 'description': 'Always apply SPF 30+ when going outdoors.', 'icon': 'wb_sunny'},
            {'title': 'Skincare Routine', 'description': 'Maintain your current healthy skin habits.', 'icon': 'shield'}
        ]
    }
}

class_labels = {
    0: 'Acne',
    1: 'Hairloss',
    2: 'Nail Fungus',
    3: 'Normal',
    4: 'Skin Allergy'
}

def predict_skin_disease(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        # Apply softmax to get confidence
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    label = class_labels.get(predicted.item(), 'Unknown')
    conf_value = confidence.item() * 100
    
    return label, conf_value

# --- API Endpoints ---

@app.route('/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    user_id = request.form.get('uid')
    image_url = request.form.get('image_url') # If already uploaded to Cloudinary
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    uploads_dir = os.path.join(BASE_DIR, 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        
    file_path = os.path.join(uploads_dir, file.filename)
    file.save(file_path)
    
    try:
        label, confidence = predict_skin_disease(file_path)
        details = DISEASE_DETAILS.get(label, {})
        
        # Save to Firestore if database is available and uid is provided
        if db and user_id:
            try:
                db.collection('user').document(user_id).collection('scan_history').add({
                    'disease_name': label,
                    'confidence': int(confidence),
                    'image_url': image_url or '',
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                print(f"History saved for user {user_id}")
            except Exception as e:
                print(f"Error saving to Firestore: {e}")

        return jsonify({
            'prediction': label,
            'confidence': confidence,
            'details': details,
            'status': 'success'
        })
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # Keep the user's requested HTML interface
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            uploads_dir = os.path.join(BASE_DIR, 'uploads')
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)
            file_path = os.path.join(uploads_dir, file.filename)
            file.save(file_path)
            label, confidence = predict_skin_disease(file_path)
            return render_template('result.html', prediction=label, confidence=f"{confidence:.2f}%")
    return render_template('upload.html')

def _get_env_bool(name, default=False):
    return str(os.environ.get(name, str(default))).lower() in ('1', 'true', 'yes')

if __name__ == '__main__':
    # Ensure uploads directory exists
    uploads_dir = os.path.join(BASE_DIR, 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    # Use port 5000 as requested
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = _get_env_bool('DEBUG', True)

    logging.basicConfig(level=logging.INFO)
    logging.info('Starting app on %s:%s (debug=%s)', host, port, debug)
    app.run(host=host, port=port, debug=debug)
