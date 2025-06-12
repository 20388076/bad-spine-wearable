#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Arduino.h>
#include <ESP32Servo.h>
#include <Wire.h>
#include "DecisionTree.h"
#include "ESP_fft.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

using namespace Eloquent::ML::Port;

Adafruit_MPU6050 mpu;
Servo myServo;
DecisionTree model;
TaskFunction_t Task1code1;
TaskFunction_t Task1code2;
TaskHandle_t Task1;
TaskHandle_t Task2;
#define FFT_SIZE 32
ESP_fft fft(FFT_REAL, FFT_SIZE, 400); // Correct FFT initialization

// Declare predict function if not already declared in DecisionTree.h
// extern "C" int predict(float* input_buffer);

// This code is for a wearable posture detection system using an ESP32, MPU6050 sensor, and a servo motor.
// Varables Initialization

int initial_position = 85;      // Start at 85 degrees
int min_position = 45;          // Minimum position of servo motor
int randomMoves = 10;           // Number of random movements
int senario = 3;                // Senario of expirament 1-3
static const int servoPin = 25; // Connect servo to pin 25 or D2
unsigned long previousMillis = 0;
int new_position = 0;                 // New position for servo motor
int last_position = initial_position; // Last position of servo motor
int step = 1;                         // Factor to decrease position by degrees per miniute
int anomaly = 0;                      // Anomaly detection flag

float input_buffer[180 + 10]; // 180 raw + 10 features
int sample_index = 0;

float gyro_y_data[30], gyro_z_data[30], acc_x_data[30], gyro_x_data[30], acc_z_data[30];

// Features extraction functions

float
compute_energy(float* data, int len) {
    float sum = 0;
    for (int i = 0; i < len; i++) {
        sum += data[i] * data[i];
    }
    return sum / len;
}

float
window_max(float* data, int len) {
    float max_val = data[0];
    for (int i = 1; i < len; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
}

float
window_mean(float* data, int len) {
    float sum = 0;
    for (int i = 0; i < len; i++) {
        sum += data[i];
    }
    return sum / len;
}

float
window_min(float* data, int len) {
    float min_val = data[0];
    for (int i = 1; i < len; i++) {
        if (data[i] < min_val) {
            min_val = data[i];
        }
    }
    return min_val;
}

float
compute_mad(float* data, int len) {
    float mean = window_mean(data, len);
    float sum = 0;
    for (int i = 0; i < len; i++) {
        sum += fabs(data[i] - mean);
    }
    return sum / len;
}

float
compute_iqr(float* data, int len) {
    std::sort(data, data + len);
    float q1 = data[len / 4];
    float q3 = data[3 * len / 4];
    return q3 - q1;
}

//Task1code: getting data from MPU6050
void
Task1code(void* pvParameters) {
    for (;;) {
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);

        input_buffer[sample_index * 6 + 0] = a.acceleration.x;
        input_buffer[sample_index * 6 + 1] = a.acceleration.y;
        input_buffer[sample_index * 6 + 2] = a.acceleration.z;
        input_buffer[sample_index * 6 + 3] = g.gyro.x;
        input_buffer[sample_index * 6 + 4] = g.gyro.y;
        input_buffer[sample_index * 6 + 5] = g.gyro.z;

        acc_x_data[sample_index] = a.acceleration.x;
        acc_z_data[sample_index] = a.acceleration.z;
        gyro_x_data[sample_index] = g.gyro.x;
        gyro_y_data[sample_index] = g.gyro.y;
        gyro_z_data[sample_index] = g.gyro.z;

        sample_index++;

        if (sample_index == 30) {
            int idx = 180;

            input_buffer[idx++] = compute_energy(gyro_y_data, 30); // ENERGY_gyro y
            input_buffer[idx++] = window_max(gyro_z_data, 30);     // gyro_z_window_max
            input_buffer[idx++] = compute_mad(acc_x_data, 30);     // MAD_acceleration x
            input_buffer[idx++] = compute_mad(gyro_y_data, 30);    // MAD_gyro y

            // === FFT Feature: FFT_gyro y ===
            for (int i = 0; i < FFT_SIZE; i++) {
                fft.real[i] = (i < 30) ? gyro_y_data[i] : 0; // zero padding
                fft.imag[i] = 0;
            }
            fft.Windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
            fft.Compute(FFT_FORWARD);
            fft.ComplexToMagnitude();
            float fft_sum = 0;
            for (int i = 1; i < 15; i++) {
                fft_sum += fft.magn[i];
            }
            input_buffer[idx++] = fft_sum;

            input_buffer[idx++] = window_mean(gyro_y_data, 30); // gyro_y_window_mean
            input_buffer[idx++] = compute_iqr(gyro_x_data, 30); // IQR_gyro x
            input_buffer[idx++] = window_min(gyro_x_data, 30);  // gyro_x_window_min
            input_buffer[idx++] = compute_mad(gyro_x_data, 30); // MAD_gyro x
            input_buffer[idx++] = compute_iqr(acc_z_data, 30);  // IQR_acceleration z

            // === Print input_buffer for Debugging ===
            Serial.println("\n=== input_buffer ===");
            for (int i = 0; i < idx; i++) {
                Serial.printf("%.3f, ", input_buffer[i]);
                if ((i + 1) % 6 == 0) {
                    Serial.println();
                }
            }

            int result = model.predict(input_buffer);
            Serial.printf("\nPrediction result: %d\n", result);

            sample_index = 0;
        }

        vTaskDelay(pdMS_TO_TICKS(100));

        /*
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp); // Acceleration is m/s^2, gyro data is rad/s //

        // Flatten and store into input_buffer
        input_buffer[sample_index * 6 + 0] = a.acceleration.x;
        input_buffer[sample_index * 6 + 1] = a.acceleration.y;
        input_buffer[sample_index * 6 + 2] = a.acceleration.z;
        input_buffer[sample_index * 6 + 3] = g.gyro.x;
        input_buffer[sample_index * 6 + 4] = g.gyro.y;
        input_buffer[sample_index * 6 + 5] = g.gyro.z;

        Serial.printf("\n%lu, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", millis(), a.acceleration.x, a.acceleration.y,
                      a.acceleration.z, g.gyro.x, g.gyro.y, g.gyro.z);
        
        sample_index++;

        // If 30 samples collected, run inference
        if (sample_index == 30) {
            int result = model.predict(input_buffer);
            Serial.printf("\nPrediction result: %d\n", result);
            sample_index = 0;  // Reset for next 30 samples
        }
        vTaskDelay(pdMS_TO_TICKS(100));
        */
    }
}

//Task2code: move servo motor randomly in a range of 0-20 degrees
void
Task2code(void* pvParameters) {
    unsigned long interval = random(5000, 120000); // Move random every 5 seconds - 2 minutes

    for (;;) {
        unsigned long currentMillis = millis();
        if (senario == 1) { // No movement detection
            // myServo.write(40);
        } else if (senario == 2) {
            if (currentMillis - previousMillis >= interval) {
                previousMillis = currentMillis; // Reset timer only after execution
                int newAngle = random(0, 21);   // Random angle between 0 and 20 degrees
                int tempPosition = initial_position + newAngle;
                myServo.write(tempPosition);      // Move to random position
                int ticks = random(900, 4000);    // Wait random from 900 ms - 2 seconds
                vTaskDelay(pdMS_TO_TICKS(ticks)); // **Use vTaskDelay instead of delay()**
                myServo.write(initial_position);  // Return to the last known position
            }
        }

        else if (senario == 3) { // Moning gradually from 85° to 45°
            if (last_position > min_position) {
                new_position = last_position - 1; // Decrease position by 1 degree
                myServo.write(new_position);      // Move the servo
                last_position = new_position;     // Update last position
                int ticks = 60000 / step;         // Calculate time in ms to wait per step
                vTaskDelay(pdMS_TO_TICKS(ticks)); // Wait milliseconds per step value

                if (last_position == min_position) {
                    last_position = initial_position; // Reset last position to initial position
                    myServo.write(initial_position);  // Return to the last known position
                }

                if (anomaly == 1) {
                    if (currentMillis - previousMillis >= interval) {
                        previousMillis = currentMillis; // Reset timer only after execution
                        int newAngle = random(0, 21);   // Random angle between 0 and 20 degrees
                        int tempPosition = last_position + newAngle;
                        myServo.write(tempPosition);      // Move to random position
                        int ticks = random(900, 4000);    // Wait random from 900 ms - 2 seconds
                        vTaskDelay(pdMS_TO_TICKS(ticks)); // **Use vTaskDelay instead of delay()**
                        myServo.write(initial_position);  // Return to the last known position
                        last_position = initial_position; // Reset last position to initial position
                    }
                }
            }
        }
    }
}

// Try to initialize servo and mpu6050!
void
setup() {
    Serial.begin(115200);
    myServo.attach(servoPin);        // Attach servo to pin 26
    myServo.write(initial_position); // Set initial position
    randomSeed(analogRead(A0));      // Seed signals from port for random numbers
    Serial.begin(115200);

    Serial.println("Wearable Posture Detection System, version:, author: ACHILLIOS PITTSILKAS");

    if (!mpu.begin()) {
        Serial.println("Failed to find MPU6050 chip");
        while (1) {
            delay(10);
        }
    }
    Serial.println("MPU6050 Found!");

    mpu.setAccelerometerRange(MPU6050_RANGE_8_G); // Setting accelerometer range of measurements to 8 G force
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

    mpu.setFilterBandwidth(MPU6050_BAND_5_HZ); // Setting mpu filter bandwidth to 5Hz
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

    // Create a task that will be executed in the Task1code() function, with priority 1 and executed on core 0
    xTaskCreatePinnedToCore(Task1code, /* Task function. */
                            "Task1",   /* name of task. */
                            10000,     /* Stack size of task */
                            NULL,      /* parameter of the task */
                            1,         /* priority of the task */
                            &Task1,    /* Task handle to keep track of created task */
                            0);        /* pin task to core 0 */
    delay(500);

    // Create a task that will be executed in the Task2code() function, with priority 1 and executed on core 1
    xTaskCreatePinnedToCore(Task2code, /* Task function. */
                            "Task2",   /* name of task. */
                            10000,     /* Stack size of task */
                            NULL,      /* parameter of the task */
                            1,         /* priority of the task */
                            &Task2,    /* Task handle to keep track of created task */
                            1);        /* pin task to core 1 */
    delay(500);
}

void
loop() {}
