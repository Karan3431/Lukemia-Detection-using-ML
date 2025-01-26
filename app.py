import os
from flask import Flask, request, jsonify, send_from_directory, url_for, current_app
from flask_cors import CORS
import cv2
import logging
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import time  # Add this import at the top

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure folders
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(CURRENT_DIR, 'images', 'uploads')
RESULT_FOLDER = os.path.join(CURRENT_DIR, 'images', 'results')
MODEL_PATH = os.path.join(CURRENT_DIR, 'model', 'model.pt')

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Initialize model
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    logger.info(f"Loading model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Add these routes to handle serving uploaded and result images
@app.route('/images/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)

@app.route('/images/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=False)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response

@app.route('/detect', methods=['POST'])
def detect():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        logger.info(f"Saved uploaded file to: {upload_path}")

        # Run detection
        results = model(upload_path)
        result = results[0]

        # Save result image
        result_filename = f"result_{int(time.time())}_{filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        
        # Save the annotated image
        annotated_img = results[0].plot()  # Get the plotted image
        cv2.imwrite(result_path, annotated_img)

        # Create full URLs for the images
        upload_url = url_for('serve_upload', filename=filename, _external=True)
        result_url = url_for('serve_result', filename=result_filename, _external=True)
        
        logger.info(f"Original image URL: {upload_url}")
        logger.info(f"Result image URL: {result_url}")

        return jsonify({
            "status": "success",
            "original_image": upload_url,
            "result_image": result_url,
            "message": "Detection completed"
        }), 200

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Routes for serving files
@app.route('/')
def home():
    return send_from_directory(CURRENT_DIR, 'detect.html')

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory(CURRENT_DIR, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)