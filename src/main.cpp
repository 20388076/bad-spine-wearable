#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Arduino.h>
#include <ESP32Servo.h>
#include <Wire.h>
#include "DecisionTree_RELIFF_FEATURES_10_Best.h"
#include "ESP_fft.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define FFT_N 32 // Must be a power of 2 >= 30

float FFT_input[FFT_N];
float FFT_output[FFT_N];

using namespace Eloquent::ML::Port;

Adafruit_MPU6050 mpu;
Servo myServo;
DecisionTree model;
TaskHandle_t Task1, Task2;
TaskFunction_t Task1code1, Task1code2;

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

float input_buffer[180];
int sample_index = 0;

float acc_x_data[32], acc_y_data[32], gyro_y_data[32], gyro_z_data[32], gyro_x_data[32], acc_z_data[32];

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

void
Task1code(void* pvParameters) {
    for (;;) {
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);
        sample_index = 0;

        for (int i = 0; i < 32; i++) {

            acc_x_data[i] = a.acceleration.x;
            acc_y_data[i] = a.acceleration.y;
            acc_z_data[i] = a.acceleration.z;
            gyro_x_data[i] = g.gyro.x;
            gyro_y_data[i] = g.gyro.y;
            gyro_z_data[i] = g.gyro.z;
            sample_index = i;
        }

        /*
    Serial.println("\n=== data ===\n");
    for (int i = 0; i < 32; i++) {
        Serial.printf("%.3f\t", acc_x_data[i]);
        Serial.printf("%.3f\t", acc_y_data[i]);
        Serial.printf("%.3f\t", acc_z_data[i]);
        Serial.printf("%.3f\t", gyro_x_data[i]);
        Serial.printf("%.3f\t", gyro_y_data[i]);
        Serial.printf("%.3f\t", gyro_z_data[i]);
        Serial.println();
    }
    */
        for (int i = 0; i < 32; i++) {
            FFT_input[i] = gyro_y_data[i];
        }
        ESP_fft fft(FFT_N, 100, FFT_REAL, FFT_FORWARD, FFT_input, FFT_output);
        fft.execute();

        if (sample_index == 31) {

            float output_matrix[32][10];
            float f0 = 0, f1 = 0, f2 = 0, f3 = 0, f4[FFT_N], f5 = 0, f6 = 0, f7 = 0, f8 = 0, f9 = 0;

            f0 = compute_energy(FFT_output, 32); // Energy of gyro_y
            f1 = window_max(gyro_z_data, 32);    // Maximum value of gyro_z
            f2 = compute_mad(acc_x_data, 32);    // Mean Absolute Deviation of acc_x
            f3 = compute_mad(gyro_y_data, 32);   // Mean Absolute Deviation of gyro_y
            //f4 = FFT_output;                     // FFT_gyro y
            f5 = window_mean(gyro_y_data, 32); // Mean of gyro_y
            f6 = compute_iqr(gyro_x_data, 32); // Interquartile Range of gyro_x
            f7 = window_min(gyro_x_data, 32);  // Minimum value of gyro_x
            f8 = compute_mad(gyro_x_data, 32); // Mean Absolute Deviation of gyro_x
            f9 = compute_iqr(acc_z_data, 32);  // Interquartile Range of acc_z

            // Fill matrix
            for (int i = 0; i < 32; i++) {

                output_matrix[i][0] = f0;            // Energy of gyro_y
                output_matrix[i][1] = f1;            // Maximum value of gyro_z
                output_matrix[i][2] = f2;            // Mean Absolute Deviation of acc_x
                output_matrix[i][3] = f3;            // Mean Absolute Deviation of gyro_y
                output_matrix[i][4] = FFT_output[i]; // FFT_gyro y
                output_matrix[i][5] = f5;            // Mean of gyro_y
                output_matrix[i][6] = f6;            // Interquartile Range of gyro_x
                output_matrix[i][7] = f7;            // Minimum value of gyro_x
                output_matrix[i][8] = f8;            // Mean Absolute Deviation of gyro_x
                output_matrix[i][9] = f9;            // Interquartile Range of acc_z
            }
            /*
        Serial.println("\n=== Output Matrix (32x10) ===");
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 10; j++) {
                Serial.printf("%.3f\t", output_matrix[i][j]);
            }
            Serial.println();
        }
        Serial.println("\n=== FFT_output ===");
        for (int i = 0; i < FFT_N; i++) {
            Serial.printf("%.3f\t", FFT_output[i]);
            
        }
        */
            int result = model.predict(input_buffer);
            Serial.printf("\nPrediction result: %d\n", result);
        }

        vTaskDelay(pdMS_TO_TICKS(100));
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
