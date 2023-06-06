# Import libraries
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import os
import numpy as np
import PIL

# Configure TensorFlow session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Define constants
UPLOAD_FOLDER = "storage/"
ALLOWED_EXTENSION = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024

# Configure Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Load model
alpaca_model = tf.keras.saving.load_model('effnev2tb3-alpaca.h5')
alpaca_model.predict(np.zeros((1, 300, 300, 3)))  # warm up model


# Define helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION


# Define function to predict image
def predict_image(model, image_filepath):
    # Preprocess the image
    with PIL.Image.open(image_filepath) as pred_image:
        pred_image = pred_image.resize((300, 300))
        pred_image = np.array(pred_image)
        pred_image = np.expand_dims(pred_image, axis=0)

    # Make predictions
    predictions = model.predict(pred_image)
    alpaca_rate = predictions[0][0]
    not_alpaca_rate = predictions[0][1]

    # Convert probabilities to class labels
    predicted_class = np.argmax(predictions[0])
    class_labels = ['alpaca', 'not_alpaca']
    predicted_label = class_labels[predicted_class]

    result = {
        "predicted_label": predicted_label,
        "probabilities": {
            "alpaca_rate": str(alpaca_rate),
            "not_alpaca_rate": str(not_alpaca_rate),
        }
    }

    return result


# Define Flask route for image upload
@app.route('/', methods=['POST'])
def upload_file():
    # Check if raw_image is in request files
    if 'raw_image' not in request.files:
        return jsonify({"message": "no file in request!"}, 400)

    # Check if image_file is empty
    image_file = request.files['raw_image']
    if image_file.content_length > MAX_CONTENT_LENGTH:
        return jsonify({"message": "file size is too large!"}, 400)

    # Check if image_file is an allowed file type
    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(image_file.filename)
        image_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_filepath)

        # Predict image
        result = predict_image(alpaca_model, image_filepath)

        # Remove image file
        os.remove(image_filepath)

        return jsonify({
            'message': 'file ' + filename + ' has been predicted!',
            'result': result
        }, 200)


# Run Flask server
if __name__ == "__main__":
    app.run()
