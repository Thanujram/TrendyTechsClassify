import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('/content/drive/MyDrive/Project/Sinhala Character Recognition/sinhala_character_model_v2.(200e(88)-17.10).h5')

class_labels = ['අ', 'ආ', 'ඉ', 'උ', 'ඌ', 'එ', 'ඔ', 'ක', 'කි', 'කෙ', 'ඛ', 'ග', 'ගා', 'ගි', 'ගො', 'ඝ', 'ච', 'චෙ', 'ඡ', 'ජ', 'ජෙ', 'ඣ', 'ඣි', 'ට', 'ඨ', 'ඩ', 'ඩි', 'ඪ', 'ණ', 'ණි', 'ණෙ', 'ත', 'ති', 'තු', 'තෙ', 'ථ', 'ථෙ', 'ද', 'දි', 'දු', 'දෙ', 'ධ', 'න', 'නි', 'නු', 'නෙ', 'ප', 'පි', 'පු', 'පෙ', 'ඵ', 'බ', 'බි', 'බු', 'බො', 'භ', 'ම', 'මි', 'ය', 'යෙ', 'ර', 'රු', 'ල', 'ලෙ', 'ලො', 'ව', 'වි', 'වෙ', 'ශ', 'ශි', 'ශු', 'ශෙ', 'ශො', 'ෂ', 'ස', 'හ', 'හා', 'ළ']

def predictClass(image_file):
  try:
    img = image.img_to_array(image_file)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    
    predictions = model.predict(img)

    predicted_class_index = np.argmax(predictions)

    probability_score = predictions[0][predicted_class_index]

    threshold = 0.8

    if probability_score >= threshold:
      predicted_class_label = class_labels[predicted_class_index]
      return(predicted_class_label)
    else:
      return("Out of class")
  except Exception as e:
    return("INTERNAL ERROR :", e.args)

from flask import Flask,request, jsonify
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({"Hello":"response1"})

@app.route('/classify', methods=['GET', 'POST'])
def classify_image():
    if request.method == 'POST':
      try:
        f = request.files['file']
        image = Image.open(f)
        target_size = (224, 224)
        image = image.resize(target_size)
        predicted_val = predictClass(image)
        result = {"Predicted_Val": predicted_val}
        print(result)
        return jsonify(result)
      except Exception as e:
        return jsonify("INTERNAL ERROR :", e.args)


if __name__ == '__main__':
    app.run()

# END