from flask import Flask, request, render_template, redirect
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import logging

# Base directory for reliable relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Number of Decease goes here =>
num_classes = 5

# Loading... the trained model =>
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model_path = os.path.join(BASE_DIR, 'models', 'skin_disease_model.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Defining transform =>
img_width, img_height = 150, 150
transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_skin_disease(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
    class_labels = {
        0: 'Acne',
        1: 'Hairloss',
        2: 'Nail Fungus',
        3: 'Normal',
        4: 'Skin Allergy'
    }
    return class_labels[predicted.item()]


@app.route('/', methods=['GET', 'POST'])
def upload_file():
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
            prediction = predict_skin_disease(file_path)
            return render_template('result.html', prediction=prediction)
    return render_template('upload.html')


def _get_env_bool(name, default=False):
    return str(os.environ.get(name, str(default))).lower() in ('1', 'true', 'yes')


if __name__ == '__main__':
    # Ensure uploads directory exists in the project folder
    uploads_dir = os.path.join(BASE_DIR, 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    # Configurable host/port/debug via environment variables
    # Default to 0.0.0.0 so the server is reachable on localhost and the host IP
    host = os.environ.get('HOST', os.environ.get('FLASK_RUN_HOST', '0.0.0.0'))
    port = int(os.environ.get('PORT', os.environ.get('FLASK_RUN_PORT', 5000)))
    debug = _get_env_bool('DEBUG', True if host in ('127.0.0.1', 'localhost') else False)

    logging.basicConfig(level=logging.INFO)
    logging.info('Starting app on %s:%s (debug=%s)', host, port, debug)

    # When deploying with a WSGI server (gunicorn/uwsgi), it will import `app`.
    # For local runs or simple hosting, use the built-in server configured by env vars.
    app.run(host=host, port=port, debug=debug)
