from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Get the absolute path to your application's directory
#app_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to model.keras
#model_file_path = os.path.join(app_directory, "/home/deepak04/mysite/plant_disease_prediction.h5")

# Load the trained Keras model without custom objects
#model_keras = tf.keras.models.load_model(model_file_path)


# Specify the directory and model file name
directory = ''
model_filename = 'plant_disease_prediction.h5'

# Construct the full filepath
model_filepath = os.path.join(directory, model_filename)
print(model_filepath)
# Load the Keras model
model_keras = tf.keras.models.load_model(model_filepath)


output_class = ["corn maize_northen leaf blight","Corn(maize)_Cercospora_leaf_spot","Orange", "Peach_bacterial_spot", "Peach_healthy", "Pepper bell bacterial spot", "pepper bell healthy", "potato early blight", "potato helathy", "potato late blight","rasberry healthy","squash powdery mildew","strawberry leaf scorch","strawberry healthy","Tomato bacterial spot","Tomato early blight","Tomato healthy","Tomato late blight","Tomato leaf mold","Tomato mosaic virus","Tomato septoria leaf spot","Tomato spider mites","Tomato target spot","Tomato yellow leaf curl virus","apple scab","apple blac rot","apple cedar apple rust","apple helathy","blueberry healthy","cherry(including sour) powder mildew","cherry(including sour)healthy","corn maize common rust","corn maize healthy","grape esca","grape black rot","grape healthy","grape leaf blight","soyabean healthy"]



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image
    img = request.files['image']

    if img.filename == '':
        return render_template('index.html', error="No file selected")

    img_path = "static/uploaded_image.jpg"
    img.save(img_path)

    test_image = image.load_img(img_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)

    predicted_array = model_keras.predict(test_image)
    predicted_value = output_class[np.argmax(predicted_array)]
    predicted_accuracy = round(np.max(predicted_array) * 100, 2)



    return render_template('index.html', prediction_value=predicted_value,
                               prediction_accuracy=predicted_accuracy,image_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)
