#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

#define CAMERA_MODEL_XIAO_ESP32S3
#include "camera_pins.h"

const char* ssid = "***********"; //Cambialo por el nombre de tu red
const char* password = "**********"; //Cambialo por la contraseña de tu red
const char* serverUrl = "http://192.168.100.25:8080/predict";// Dirección de tu endpoint

#define PIN_IMPRESORA 4
unsigned long lastCaptureTime = 0;
const unsigned long interval = 30 * 1000;  // 5 minutos reales

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("⏳ Iniciando...");

  pinMode(PIN_IMPRESORA, INPUT);

  WiFi.begin(ssid, password);
  Serial.print("🔌 Conectando a WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi conectado");
  Serial.print("🛰️ Dirección IP: ");
  Serial.println(WiFi.localIP());

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  Serial.println("🎥 Iniciando cámara...");
  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("❌ Error al iniciar la cámara");
    return;
  }
  Serial.println("✅ Cámara inicializada correctamente");
}

void loop() {
  unsigned long currentTime = millis();
  if (currentTime - lastCaptureTime > interval) {
    lastCaptureTime = currentTime;
    enviarFoto();
  }

  delay(1000);
}

void enviarFoto() {
  Serial.println("📸 Capturando imagen...");
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("❌ Error al capturar imagen");
    return;
  }

  Serial.printf("📦 Tamaño imagen: %d bytes\n", fb->len);

  WiFiClient client;
  if (!client.connect("192.168.100.25", 8080)) {
    Serial.println("❌ No se pudo conectar al servidor");
    esp_camera_fb_return(fb);
    return;
  }

  String boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
  String startRequest = "--" + boundary + "\r\n";
  startRequest += "Content-Disposition: form-data; name=\"file\"; filename=\"capture.jpg\"\r\n";
  startRequest += "Content-Type: image/jpeg\r\n\r\n";

  String endRequest = "\r\n--" + boundary + "--\r\n";

  int contentLength = startRequest.length() + fb->len + endRequest.length();

  // Construye cabecera HTTP completa
  client.println("POST /predict HTTP/1.1");
  client.println("Host: 192.168.100.25:8080");
  client.println("Content-Type: multipart/form-data; boundary=" + boundary);
  client.println("Content-Length: " + String(contentLength));
  client.println();  // Línea vacía para separar headers del body

  // Enviar partes del body
  client.print(startRequest);
  client.write(fb->buf, fb->len);
  client.print(endRequest);

  Serial.println("📡 Esperando respuesta...");
  while (client.connected()) {
    String line = client.readStringUntil('\n');
    if (line == "\r") break;
    Serial.println(line);  // Mostrar headers de respuesta
  }

  String body = client.readString();
  Serial.println("🧠 Respuesta:");
  Serial.println(body);

  esp_camera_fb_return(fb);
  client.stop();
}
