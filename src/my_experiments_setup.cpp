/*****************************************************
 *  Include Required Libraries
 *  These libraries let the ESP32 communicate with 
 *  the MPU6050 sensor, control the servo motor,
 *  and use FreeRTOS for multitasking.
 *****************************************************/
#include <Adafruit_MPU6050.h>  // Library for MPU6050 accelerometer + gyroscope
#include <Adafruit_Sensor.h>   // Unified sensor library used by Adafruit sensors
#include <Arduino.h>           // Core Arduino functions
#include <ESP32Servo.h>        // Library to control servo motors on ESP32
#include <Wire.h>              // I2C communication (used by MPU6050)
#include <algorithm>           // Useful for math/array operations
#include <iostream>            // Input/output (mainly for debugging with Serial)
#include "fft.h"               // Fast Fourier Transform Custom library
#include "freertos/FreeRTOS.h" // FreeRTOS real-time operating system
#include "freertos/task.h"     // FreeRTOS task handling (multithreading)

/*****************************************************
 *  Hardware and Task Declarations
 *  Here we create objects for the MPU6050 sensor, 
 *  the servo motor, and FreeRTOS task handles.
 *****************************************************/

Adafruit_MPU6050 mpu; // Object to talk with the MPU6050 sensor
Servo myServo;        // Object to control the servo motor

// Task handles let us run two different pieces of code in "parallel"
// (FreeRTOS allows multitasking on ESP32)
TaskHandle_t Task1, Task2;
TaskFunction_t Task1code1, Task1code2;

// This code is for a wearable posture detection system using an ESP32, MPU6050 sensor, and a servo motor.
// Varables Initialization

/* User configuration */

// Senario of expirament
// 1: No movement detection
// 2: Random movement detection
// 3: Gradual movement detection 1 step per minute
// 4: Gradual movement detection 2 steps per minute
// 5: Gradual movement detection 3 steps per minute with random movement / anomaly detection
int scenario = 1; // scenario of expirament 1-5

/* Servo Variables configuration */

int initial_position = 85;      // Start at 85 degrees
int min_position = 45;          // Minimum position of servo motor
static const int servoPin = 25; // Connect servo to pin 25 or D2
unsigned long previousMillis = 0;
int new_position = 0;                 // New position for servo motor
int last_position = initial_position; // Last position of servo motor
int step;                             // Factor to decrease position by degrees per miniute

/* Sampling Variables configuration */

float t;                                         // Measurements computation time variable
unsigned long start;                             // Start time variable
float sampleRate = 9.71;                         // Sample rate in Hz     <-- Change this value to set sample rate
float samplePeriod = round(1000.0 / sampleRate); // Sample period in ms

// Task1code: Read data from MPU6050 and print to Serial Monitor
void
Task1code(void* pvParameters) {
    for (;;) {
        start = millis();
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp); // Acceleration is m/s^2, gyro data is rad/s
        Serial.printf("\n%lu, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", millis(), a.acceleration.x, a.acceleration.y,
                      a.acceleration.z, g.gyro.x, g.gyro.y, g.gyro.z);
        t = millis() - start; // Calculate measurements computation time
        vTaskDelay(pdMS_TO_TICKS(
            samplePeriod
            - t)); // If e.g sample is at 50 Hz (every 1000/50 = 20 ms - processing time) wait to achieve 50 Hz
    }
}

//Task2code: Move servo motor based on scenario selected
void
Task2code(void* pvParameters) {
    unsigned long interval = random(5000, 120000); // Move random every 5 seconds - 2 minutes

    for (;;) {
        unsigned long currentMillis = millis();
        if (scenario == 1) {   // No movement detection
            myServo.write(90); // Move to neutral position
        } else if (scenario == 2) {
            if (currentMillis - previousMillis >= interval) {
                previousMillis = currentMillis;                 // Reset timer only after execution
                int newAngle = random(0, 21);                   // Random angle between 0 and 20 degrees
                int tempPosition = initial_position + newAngle; // Calculate new position
                myServo.write(tempPosition);                    // Move to random position
                int ticks = random(900, 4000);                  // Wait random from 900 ms - 2 seconds
                vTaskDelay(
                    pdMS_TO_TICKS(ticks)); // **Use vTaskDelay instead of delay() on multiprocessing** wait milliseconds
                myServo.write(initial_position); // Return to the last known position
            }
        }

        else if (scenario == 3 || 4 || 5) { // Moning gradually from 85° to 45° with servo steps per minute
            if (scenario == 4) {
                step = 2; // Decrease position by 2 servo steps per minute
            } else if (scenario == 5) {
                step = 3; // Decrease position by 3 servo steps per minute
            } else {
                step = 1; // Decrease position by 1 servo steps per minute
            }
            if (last_position > min_position) {
                new_position = --last_position;   // Decrease position by 1 servo step
                myServo.write(new_position);      // Move the servo
                last_position = new_position;     // Update last position
                int ticks = 60000 / step;         // Calculate time in ms to wait per step
                vTaskDelay(pdMS_TO_TICKS(ticks)); // Wait milliseconds per step value

                if (last_position == min_position) {
                    last_position = initial_position; // Reset last position to initial position
                    myServo.write(initial_position);  // Return to the last known position
                }

                if (scenario == 5) { // Anomaly detection and random movement
                    if (currentMillis - previousMillis >= interval) {
                        previousMillis = currentMillis;              // Reset timer only after execution
                        int newAngle = random(0, 21);                // Random angle between 0 and 20 degrees
                        int tempPosition = last_position + newAngle; // Add random angle to last position
                        myServo.write(tempPosition);                 // Move to random position
                        int ticks = random(900, 4000);               // Wait random from 900 ms - 2 seconds
                        vTaskDelay(pdMS_TO_TICKS(ticks));            // Wait milliseconds
                        myServo.write(initial_position);             // Return to the last known position
                        last_position = initial_position;            // Reset last position to initial position
                    }
                }
            }
        }
    }
}

// Try to initialize servo and mpu6050!
void
setup() {
    Serial.printf("senario: %d\n", scenario);
    Serial.begin(115200);     // Start serial communication at 115200 baud rate
    myServo.attach(servoPin); // Attach servo to pin 25 or D2
    if (scenario == 1) {
        myServo.write(90); // Set neutral position for no movement detection
    } else {
        myServo.write(initial_position); // Set initial position
    }
    randomSeed(analogRead(A0)); // Seed signals from port for random numbers

    Serial.println("Wearable Posture Detection System, version:, author: ACHILLIOS PITTSILKAS");

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