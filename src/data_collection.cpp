#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

const char* ssid = "ssid-here";
const char* password = "password-here";
const char* udpAddress = "10.178.91.19"; //  PC's IP address (ipconfig on cmd)
const int udpPort = 12345;

WiFiUDP udp;
char packetBuffer[256];

float t;                                         // Measurements computation time variable
unsigned long start;                             // Start time variable
float sampleRate = 9.71f;                        // Sample rate in Hz     <-- Change this value to set sample rate
float samplePeriod = round(1000.0 / sampleRate); // Sample period in ms

void
setup() {
    Serial.begin(115200);

    Serial.println("Wearable Posture Detection System, version:, author: ACHILLIOS PITTSILKAS");

    // WiFi Setup
    Serial.println("\nConnecting to WiFi...");
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected!");
    Serial.print("ESP32 IP address: ");
    Serial.println(WiFi.localIP());
    Serial.printf("Sending data to: %s:%d\n", udpAddress, udpPort);

    // MPU6050 Setup using Adafruit Library
    if (!mpu.begin()) {
        Serial.println("Failed to find MPU6050 chip"); // MPU6050 is not connected
        while (1) {
            delay(10);
        }
    }
    Serial.println("MPU6050 Found!"); // MPU6050 is connected

    // Set up the accelerometer, gyro ranges and filter bandwidth based on: https://adafruit.github.io/Adafruit_MPU6050/html/class_adafruit___m_p_u6050.html#a1583d1351bb907d3823aee36af0efe5f
    mpu.setAccelerometerRange(
        MPU6050_RANGE_8_G); // Setting accelerometer range of measurements to 8 G maximum acceleration it can be detected
    Serial.print("Accelerometer range set to: ");
    switch (mpu.getAccelerometerRange()) {
        case MPU6050_RANGE_2_G: Serial.println("+-2G"); break;
        case MPU6050_RANGE_4_G: Serial.println("+-4G"); break;
        case MPU6050_RANGE_8_G: Serial.println("+-8G"); break;
        case MPU6050_RANGE_16_G: Serial.println("+-16G"); break;
    }

    mpu.setGyroRange(MPU6050_RANGE_500_DEG); // Setting gyro range of measurements to 500 degrees per second
    Serial.print("Gyro range set to: ");
    switch (mpu.getGyroRange()) {
        case MPU6050_RANGE_250_DEG: Serial.println("+- 250 deg/s"); break;
        case MPU6050_RANGE_500_DEG: Serial.println("+- 500 deg/s"); break;
        case MPU6050_RANGE_1000_DEG: Serial.println("+- 1000 deg/s"); break;
        case MPU6050_RANGE_2000_DEG: Serial.println("+- 2000 deg/s"); break;
    }

    mpu.setFilterBandwidth(MPU6050_BAND_5_HZ); // Setting mpu low pass filter bandwidth to 5Hz
    Serial.print("Filter bandwidth set to: ");
    switch (mpu.getFilterBandwidth()) {
        case MPU6050_BAND_260_HZ: Serial.println("260 Hz"); break;
        case MPU6050_BAND_184_HZ: Serial.println("184 Hz"); break;
        case MPU6050_BAND_94_HZ: Serial.println("94 Hz"); break;
        case MPU6050_BAND_44_HZ: Serial.println("44 Hz"); break;
        case MPU6050_BAND_21_HZ: Serial.println("21 Hz"); break;
        case MPU6050_BAND_10_HZ: Serial.println("10 Hz"); break;
        case MPU6050_BAND_5_HZ: Serial.println("5 Hz"); break;
    }

    Serial.println("");
    delay(100);
}

void
loop() {
    start = millis();
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp); // Acceleration is m/s^2, gyro data is rad/s

    snprintf(packetBuffer, sizeof(packetBuffer), "%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f", millis(), a.acceleration.x,
             a.acceleration.y, a.acceleration.z, g.gyro.x, g.gyro.y, g.gyro.z);
    // Send UDP packet to PC
    udp.beginPacket(udpAddress, udpPort);
    udp.write((uint8_t*)packetBuffer, strlen(packetBuffer));
    udp.endPacket();

    // Also print to Serial for debugging
    Serial.printf("\n%s", packetBuffer);

    t = millis() - start; // Calculate measurements computation time
    vTaskDelay(pdMS_TO_TICKS(
        samplePeriod - t)); // If e.g sample is at 50 Hz (every 1000/50 = 20 ms - processing time) wait to achieve 50 Hz
}

/*

static float scratch1[WINDOW];
static float scratch2[WINDOW];
// -----------------------------------------------------------------------

int sample_index = 0;
const float G_CONST = 9.80665f;
float samplePeriod = round(1000.0 / sampleRate);
float t1, t2;
unsigned long start;
float theta_x, theta_y, theta_z;

// ---- Mean ----
float window_mean(float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += data[i];
    return sum / n;
}

float window_max(float* data, int n) {
    float m = data[0];
    for (int i = 1; i < n; i++) if (data[i] > m) m = data[i];
    return m;
}

float window_min(float* data, int n) {
    float m = data[0];
    for (int i = 1; i < n; i++) if (data[i] < m) m = data[i];
    return m;
}

float compute_rms(float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += data[i] * data[i];
    return sqrt(sum / n);
}

float compute_var(float* data, int n) {
    float m = window_mean(data, n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = data[i] - m;
        sum += d * d;
    }
    return sum / n;
}

float compute_std(float* data, int n) { return sqrt(compute_var(data, n)); }

float compute_mad(float* data, int n) {
    float m = window_mean(data, n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += fabs(data[i] - m);
    return sum / n;
}

float compute_iqr(float* data, int len) {
    for (int i = 0; i < len; ++i) scratch1[i] = data[i];
    std::sort(scratch1, scratch1 + len);
    float q1_pos = (len - 1) * 0.25f;
    float q3_pos = (len - 1) * 0.75f;
    int q1_idx = (int)q1_pos;
    int q3_idx = (int)q3_pos;
    float q1 = scratch1[q1_idx] + (scratch1[q1_idx + 1] - scratch1[q1_idx]) * (q1_pos - q1_idx);
    float q3 = scratch1[q3_idx] + (scratch1[q3_idx + 1] - scratch1[q3_idx]) * (q3_pos - q3_idx);
    return q3 - q1;
}

float compute_sma_median(float* ax, float* ay, float* az, int n) {
    for (int i = 0; i < n; i++) scratch1[i] = fabs(ax[i]) + fabs(ay[i]) + fabs(az[i]);
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (scratch1[j] > scratch1[j + 1]) {
                float tmp = scratch1[j];
                scratch1[j] = scratch1[j + 1];
                scratch1[j + 1] = tmp;
            }
        }
    }
    return scratch1[n / 2];
}

float compute_median(float* data, int n) {
    for (int i = 0; i < n; ++i) scratch1[i] = data[i];
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (scratch1[j] > scratch1[j + 1]) {
                float tmp = scratch1[j];
                scratch1[j] = scratch1[j + 1];
                scratch1[j + 1] = tmp;
            }
        }
    }
    if (n % 2 == 0) return (scratch1[n / 2 - 1] + scratch1[n / 2]) / 2.0f;
    else return scratch1[n / 2];
}

float vector_magnitude(float* x, float* y, float* z, int n) {
    for (int i = 0; i < n; ++i) scratch1[i] = sqrtf(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
    return compute_median(scratch1, n);
}

float cubic_prod_median(float* x, float* y, float* z, int n) {
    for (int i = 0; i < n; ++i) {
        float prod = fabsf(x[i]*y[i]*z[i]);
        scratch1[i] = powf(prod, 1.0f/3.0f);
    }
    return compute_median(scratch1, n);
}



 */
