from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import logging
import io

# Configura logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar modelo
logger.info("Cargando modelo...")
model = load_model("printing_model.h5")
IMG_SIZE = (224, 224)
class_names = ['OK', 'blobs', 'cracks', 'spaghetti', 'stringing', 'under exstrosion']
logger.info("Modelo cargado exitosamente.")

# Crear app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    logger.info("Solicitud recibida en /predict")
    
    if "file" not in request.files:
        logger.warning("No se recibió archivo en la solicitud.")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    logger.info(f"Archivo recibido: {file.filename}")

    try:
        if "file" in request.files:
            file = request.files["file"]
            logger.info(f"Archivo recibido: {file.filename}")
            image_bytes = file.read()
        else:
            logger.info("Archivo recibido como raw bytes (no multipart/form-data)")
            image_bytes = request.get_data()

        image_stream = io.BytesIO(image_bytes)
        img = load_img(image_stream, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error procesando la imagen: {e}")
        return jsonify({"error": "Invalid image format"}), 400


    try:
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        predicted_class = class_names[class_index]
        confidence = float(np.max(prediction))

        logger.info(f"Predicción: {predicted_class} ({confidence:.4f})")

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        logger.error(f"Error durante la predicción: {e}")
        return jsonify({"error": "Prediction failed"}), 500

# Iniciar server
if __name__ == "__main__":
    logger.info("Iniciando servidor Flask en puerto 8080...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


'''
curl -X POST http://localhost:8080/predict \
  -F "file=@/path_to_your_imabe/img.png"
'''