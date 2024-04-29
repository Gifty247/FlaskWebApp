from flask import Flask, request, render_template
import os
from torch_utils import process_lane, model_inference, create_output_image
import cv2
import numpy as np

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index(name=None):
    return render_template(
        'index.html', 
        title="Lane Detection Application", 
        description="Upload an image and detect lanes in it."
        )


@app.route('/upload', methods=['POST'])
def upload_image():
    
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filepath = os.path.join('static', file.filename)
        file.save(filepath)
        image_tensor = process_lane(filepath)
        prediction = model_inference(image_tensor)
        original_image = cv2.imread(filepath)
        output_image = create_output_image(original_image, prediction)
        output_filename = 'processed_' + file.filename
        cv2.imwrite(os.path.join('static', output_filename), output_image)
        return render_template('index.html', image_path=output_filename, lanes_detected="Lanes Detected")
    else:
        return "Unsupported file type", 400

        
if __name__ == '__main__':
    app.run(debug=True)
