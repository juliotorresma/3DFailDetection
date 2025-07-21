# Clasificador de Errores de Impresión 3D con Flask + ESP32-CAM + IA

¡Bienvenido al proyecto más nerd y útil que jamás pensaste armar!  
Este sistema toma una foto con tu ESP32-CAM, la manda a una API local que corre un modelo de IA entrenado con TensorFlow, y te dice si tu impresión 3D es arte... o un desastre 🫠

---

## 🧩 ¿Qué necesitas?

### Hardware:
- 1x ESP32-CAM (o similar)
- 1x impresora 3D (idealmente una que ocasionalmente falle 😅)

### Software:
- Python 3.8+
- TensorFlow + Keras
- Flask
- curl o `requests` para pruebas
- Dataset con imágenes de errores (`cracks`, `spaghetti`, etc.)

---

## 📁 Repositorio del Proyecto

Este proyecto es de código abierto y está disponible en:

🔗 [https://github.com/juliotorresma/3DFailDetection](https://github.com/juliotorresma/3DFailDetection)

Puedes clonarlo directamente con:

```bash
git clone https://github.com/juliotorresma/3DFailDetection.git
```
---


## 📚 ¿Cómo funciona el modelo?

Este proyecto utiliza **Transfer Learning** con un modelo preentrenado llamado `MobileNetV2`.

### 🤖 ¿Qué es Transfer Learning?

Es cuando tomamos un modelo entrenado con millones de imágenes (como ImageNet) y le “enseñamos” a reconocer **nuestras clases específicas** con muy pocos datos.

### 🔢 Explicación matemática (simplificada):

El modelo realiza la siguiente operación:

```math
\hat{y} = \text{softmax}(W_2 \cdot \text{ReLU}(W_1 \cdot f(x) + b_1) + b_2)
```

Donde:

- \( x \): imagen de entrada (por ejemplo, una foto de una impresión fallida)
- \( f(x) \): representación profunda extraída por `MobileNetV2` (convoluciones + pooling)
- \( W_1, W_2 \): pesos entrenables de las capas densas finales
- \( \hat{y} \): vector de probabilidades para cada clase (`OK`, `blobs`, etc.)

El objetivo es minimizar la **pérdida de entropía cruzada**:

```math
\mathcal{L} = -\sum_{i=1}^{C} y_i \cdot \log(\hat{y}_i)
```

Donde:

- \( y \): vector one-hot de la clase real
- \( \hat{y} \): predicción del modelo
- \( C \): número de clases (en este caso, 6)

En palabras simples: el modelo trata de que la **probabilidad de la clase correcta** sea lo más cercana a 1 posible.

---

## 📦 1. Entrenando el modelo con Transfer Learning

```bash
pip install tensorflow scikit-learn matplotlib
```

Asegúrate de tener tu dataset así:

```
printing-errors/
├── OK/
├── blobs/
├── cracks/
├── spaghetti/
├── stringing/
└── under exstrosion/
```

Ahora corre este script:

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Configuración
IMG_SIZE = (224, 224)
DATA_DIR = "printing-errors"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, subset='training',
    class_mode='categorical', batch_size=8
)

val_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, subset='validation',
    class_mode='categorical', batch_size=8
)

# Modelo base
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=15)

# Guardar modelo
model.save("printing_model.h5")
```

---

## 🌐 2. Servidor Flask para clasificar imágenes

```bash
pip install flask
```

Crea un archivo llamado `api.py`:

```python
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import io, logging, os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = load_model("printing_model.h5")
IMG_SIZE = (224, 224)
class_names = ['OK', 'blobs', 'cracks', 'spaghetti', 'stringing', 'under exstrosion']

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    logger.info("Solicitud recibida de clasificar imagen")
    try:
        if "file" in request.files:
            image_bytes = request.files["file"].read()
        else:
            image_bytes = request.get_data()

        img = load_img(io.BytesIO(image_bytes), target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        idx = np.argmax(pred)
        return jsonify({
            "prediction": class_names[idx],
            "confidence": float(np.max(pred))
        })

    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
```

Correlo con:

```bash
python api.py
```

---

## 📷 3. Enviar imagen desde el ESP32-CAM

Este fragmento de Arduino manda una foto capturada al servidor Flask:

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"

const char* ssid = "TuWiFi";
const char* password = "TuPassword";
const char* server = "http://192.168.1.100:8080/predict"; // IP de tu PC

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) delay(500);
  Serial.println("Conectado!");

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Error capturando imagen");
    return;
  }

  HTTPClient http;
  http.begin(server);
  http.addHeader("Content-Type", "image/jpeg");
  int httpResponseCode = http.POST(fb->buf, fb->len);

  String response = http.getString();
  Serial.println("Respuesta: " + response);
  http.end();

  esp_camera_fb_return(fb);
}

void loop() {}
```

---

## 🧪 4. Probar manualmente con curl

```bash
curl -X POST http://192.168.1.100:8080/predict \
  -F "file=@fail01.jpg"
```

O usando raw JPEG:

```bash
curl -X POST http://192.168.1.100:8080/predict \
```

## 🎥 Video de apoyo

Mira el proyecto en acción:

[🔗 Clasificador de errores en impresión 3D con IA - YouTube](https://www.youtube.com/watch?v=qNzlytUdB_Q&t=913s)

[![Video de apoyo](https://img.youtube.com/vi/qNzlytUdB_Q/0.jpg)](https://www.youtube.com/watch?v=qNzlytUdB_Q)
