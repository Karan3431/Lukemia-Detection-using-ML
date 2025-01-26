import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Define upload and result folders
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
RESULT_FOLDER = os.path.join(os.getcwd(), 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load the YOLO model
model = YOLO('/Users/karansingh/Downloads/Yolov8(Lukemia detection).pt')  # Update this path to your YOLO model file

@app.route('/')
def home():
    """Serve the index.html file."""
    return send_from_directory(os.getcwd(), 'detect.html')

@app.route('/<page>')
def serve_page(page):
    """Serve other HTML pages."""
    try:
        return send_from_directory(os.getcwd(), f'{page}.html')
    except Exception as e:
        return f"Error: {e}", 404

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files."""
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/detect', methods=['POST'])
def detect():
    """Handle image upload and detection."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    filename = secure_filename(file.filename)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(img_path)

    try:
        # Perform detection
        results = model(img_path)

        # Save the result image
        result_img_path = os.path.join(app.config['RESULT_FOLDER'], f"result_{filename}")
        result_image = results[0].plot(show=False)
        cv2.imwrite(result_img_path, result_image)

        # Generate URLs for the uploaded and result images
        uploaded_url = f'/uploads/{filename}'
        result_url = f'/results/result_{filename}'

        return jsonify({
            "uploaded_image": uploaded_url,
            "result_image": result_url
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/<filename>')
def serve_static_file(filename):
    """Serve static files like JS and CSS."""
    return send_from_directory(os.getcwd(), filename)

if __name__ == '__main__':
    app.run(debug=True)