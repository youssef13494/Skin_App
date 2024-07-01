# from flask import Flask, request, render_template
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np

# app = Flask(__name__)

# # Load the skin disease classification model during initialization
# model = load_model(r"C:\Users\abdel\Desktop\Final_Try\skin_cancer_detection7.h5")

# # Define the classes for skin diseases
# classes = {
#     0: ('Actinic keratoses and intraepithelial carcinomae', 'Actinic keratoses and intraepithelial carcinomae'),
#     1: ('Basal cell carcinoma', 'Basal cell carcinoma'),
#     2: ('Benign keratosis-like lesions', 'Benign keratosis-like lesions'),
#     3: ('Dermatofibroma', 'Dermatofibroma'),
#     4: ('Melanoma', 'Melanoma'),
#     5: ('Melanocytic nevi', 'Melanocytic nevi'),
#     6: ('Pyogenic granulomas and hemorrhage', 'Pyogenic granulomas and hemorrhage')
# }

# # Function to preprocess the image for skin disease classification
# def preprocess_image(image):
#     image = image.resize((90, 120))  # Resize the image to match the model's input shape
#     image_array = np.array(image)  # Convert image to array
#     image_array = image_array / 255.0  # Normalize the pixel values
#     return image_array

# @app.route('/')
# def index():
#     return render_template('index.html', appName="Skin Disease Detection", prediction=None, image=None)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'fileup' not in request.files:
#         return render_template('index.html', appName="Skin Disease Detection", prediction=None, image=None, error='Please upload an image')

#     file = request.files['fileup']
#     if file.filename == '':
#         return render_template('index.html', appName="Skin Disease Detection", prediction=None, image=None, error='Please upload a valid image')

#     try:
#         image = Image.open(file)
#         image_array = preprocess_image(image)
#         # Make predictions
#         predictions = model.predict(np.array([image_array]))
#         # Interpret the predictions
#         predicted_class_index = np.argmax(predictions)
#         predicted_class_name = classes[predicted_class_index][0]
#         predicted_class_description = classes[predicted_class_index][1]
#         return render_template('index.html', appName="Skin Disease Detection", prediction=predicted_class_name, description=predicted_class_description, image=file)
#     except Exception as e:
#         return render_template('index.html', appName="Skin Disease Detection", prediction=None, image=None, error=str(e))

# if __name__ == '__main__':
#     app.run(debug=True)



###################################################################

from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the skin disease classification model during initialization
model = load_model(r"C:\Users\abdel\Desktop\Final_Try\skin_cancer_detection7.h5")

# Define the classes for skin diseases
classes = {
    0: ('Actinic keratoses and intraepithelial carcinomae', 'Actinic keratoses and intraepithelial carcinomae'),
    1: ('Basal cell carcinoma', 'Basal cell carcinoma'),
    2: ('Benign keratosis-like lesions', 'Benign keratosis-like lesions'),
    3: ('Dermatofibroma', 'Dermatofibroma'),
    4: ('Melanoma', 'Melanoma'),
    5: ('Melanocytic nevi', 'Melanocytic nevi'),
    6: ('Pyogenic granulomas and hemorrhage', 'Pyogenic granulomas and hemorrhage')
}

# Function to preprocess the image for skin disease classification
def preprocess_image(image):
    image = Image.open(image)
    image = image.resize((90, 120))  # Resize the image to match the model's input shape
    image_array = np.array(image.convert('RGB'))  # Convert image to RGB and then to array
    image_array = image_array / 255.0  # Normalize the pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/')
def index():
    return render_template('index.html', appName="Skin Disease Detection")

@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return jsonify({'error': "Please try again. The Image doesn't exist"})
        
        image = request.files.get('fileup')
        image_array = preprocess_image(image)
        print("Model predicting ...")
        result = model.predict(image_array)
        print("Model predicted")
        ind = np.argmax(result)
        prediction = classes[ind][0]
        print(prediction)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        try:
            if 'fileup' not in request.files:
                return render_template('index.html', appName="Skin Disease Detection", error='Please upload an image')
            
            # Get the image from post request
            print("image loading....")
            image = request.files['fileup']
            print("image loaded....")
            image_array = preprocess_image(image)
            print("predicting ...")
            result = model.predict(image_array)
            print("predicted ...")
            ind = np.argmax(result)
            prediction = classes[ind][0]
            description = classes[ind][1]

            print(prediction)

            return render_template('index.html', prediction=prediction, description=description, image='static/IMG/', appName="Skin Disease Detection")
        except Exception as e:
            return render_template('index.html', appName="Skin Disease Detection", error=str(e))
    else:
        return render_template('index.html', appName="Skin Disease Detection")

if __name__ == '__main__':
    app.run(debug=True)


###############################################################################

# from flask import Flask, request, render_template, jsonify
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np

# app = Flask(__name__)

# # Load the skin disease classification model during initialization
# model = load_model(r"C:\Users\abdel\Desktop\Final_Try\skin_cancer_detection7.h5")

# # Define the classes for skin diseases
# classes = {
#     0: ('Actinic keratoses and intraepithelial carcinomae', 'Actinic keratoses and intraepithelial carcinomae'),
#     1: ('Basal cell carcinoma', 'Basal cell carcinoma'),
#     2: ('Benign keratosis-like lesions', 'Benign keratosis-like lesions'),
#     3: ('Dermatofibroma', 'Dermatofibroma'),
#     4: ('Melanoma', 'Melanoma'),
#     5: ('Melanocytic nevi', 'Melanocytic nevi'),
#     6: ('Pyogenic granulomas and hemorrhage', 'Pyogenic granulomas and hemorrhage')
# }

# # Function to preprocess the image for skin disease classification
# def preprocess_image(image):
#     image = Image.open(image)
#     image = image.resize((90, 120))  # Resize the image to match the model's input shape
#     image_array = np.array(image.convert('RGB'))  # Convert image to RGB and then to array
#     image_array = image_array / 255.0  # Normalize the pixel values
#     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
#     return image_array

# @app.route('/')
# def index():
#     return render_template('index.html', appName="Skin Disease Detection")

# @app.route('/predictApi', methods=["POST"])
# def api():
#     # Get the image from post request
#     try:
#         if 'fileup' not in request.files:
#             return jsonify({'error': "Please try again. The Image doesn't exist"})
        
#         image = request.files.get('fileup')
#         image_array = preprocess_image(image)
#         print("Model predicting ...")
#         result = model.predict(image_array)
#         print("Model predicted")
#         ind = np.argmax(result)
#         prediction = classes[ind][0]
#         print(prediction)
#         return jsonify({'prediction': prediction})
#     except Exception as e:
#         return jsonify({'error': str(e)})

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             if 'fileup' not in request.files:
#                 return render_template('index.html', appName="Skin Disease Detection", error='Please upload an image')

#             # Get the image from post request
#             image = request.files['fileup']
#             image_array = preprocess_image(image)
#             result = model.predict(image_array)
#             ind = np.argmax(result)
#             prediction = classes[ind][0]
#             description = classes[ind][1]

#             return render_template('index.html', prediction=prediction, description=description, appName="Skin Disease Detection")
#         except Exception as e:
#             return render_template('index.html', appName="Skin Disease Detection", error=str(e))
#     else:
#         return render_template('index.html', appName="Skin Disease Detection")

# # if __name__ == '__main__':
# #     app.run(debug=True)
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080, debug=False)

########################################################################################


# from flask import Flask, request, render_template, jsonify
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np
# from flask_ngrok import run_with_ngrok

# app = Flask(__name__)
# run_with_ngrok(app)  # This will start ngrok automatically when you run the app

# # Load the skin disease classification model during initialization
# model = load_model(r"C:\Users\abdel\Desktop\Final_Try\skin_cancer_detection7.h5")

# # Define the classes for skin diseases
# classes = {
#     0: ('Actinic keratoses and intraepithelial carcinomae', 'Actinic keratoses and intraepithelial carcinomae'),
#     1: ('Basal cell carcinoma', 'Basal cell carcinoma'),
#     2: ('Benign keratosis-like lesions', 'Benign keratosis-like lesions'),
#     3: ('Dermatofibroma', 'Dermatofibroma'),
#     4: ('Melanoma', 'Melanoma'),
#     5: ('Melanocytic nevi', 'Melanocytic nevi'),
#     6: ('Pyogenic granulomas and hemorrhage', 'Pyogenic granulomas and hemorrhage')
# }

# # Function to preprocess the image for skin disease classification
# def preprocess_image(image):
#     image = Image.open(image)
#     image = image.resize((90, 120))  # Resize the image to match the model's input shape
#     image_array = np.array(image.convert('RGB'))  # Convert image to RGB and then to array
#     image_array = image_array / 255.0  # Normalize the pixel values
#     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
#     return image_array

# @app.route('/')
# def index():
#     return render_template('index.html', appName="Skin Disease Detection")

# @app.route('/predictApi', methods=["POST"])
# def api():
#     # Get the image from post request
#     try:
#         if 'fileup' not in request.files:
#             return jsonify({'error': "Please try again. The Image doesn't exist"})
        
#         image = request.files.get('fileup')
#         image_array = preprocess_image(image)
#         print("Model predicting ...")
#         result = model.predict(image_array)
#         print("Model predicted")
#         ind = np.argmax(result)
#         prediction = classes[ind][0]
#         print(prediction)
#         return jsonify({'prediction': prediction})
#     except Exception as e:
#         return jsonify({'error': str(e)})

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             if 'fileup' not in request.files:
#                 return render_template('index.html', appName="Skin Disease Detection", error='Please upload an image')

#             # Get the image from post request
#             image = request.files['fileup']
#             image_array = preprocess_image(image)
#             result = model.predict(image_array)
#             ind = np.argmax(result)
#             prediction = classes[ind][0]
#             description = classes[ind][1]

#             return render_template('index.html', prediction=prediction, description=description, appName="Skin Disease Detection")
#         except Exception as e:
#             return render_template('index.html', appName="Skin Disease Detection", error=str(e))
#     else:
#         return render_template('index.html', appName="Skin Disease Detection")

# if __name__ == '__main__':
#     app.run()
