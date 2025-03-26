#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Arduino.h>
#include <ESP32Servo.h>
#include <Wire.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

int position = 85;              // Start at 85 degrees
int randomMoves = 10;           // Number of random movements
int senario = 2;                // Senario of expirament
static const int servoPin = 25; // Connect servo to pin 25 or D2
unsigned long previousMillis = 0;
int new_position = 0;         // New position for servo motor
int last_position = position; // Last position of servo motor
int step = 1;                 // Factor to decrease position by 1 degree

Adafruit_MPU6050 mpu;
Servo myServo;

TaskFunction_t Task1code1;
TaskFunction_t Task1code2;
TaskHandle_t Task1;
TaskHandle_t Task2;

//Task1code: getting data from MPU6050
void
Task1code(void* pvParameters) {
    for (;;) {
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp); /* Acceleration is m/s^2, gyro data is rad/s */
        Serial.printf("\n%lu, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", millis(), a.acceleration.x, a.acceleration.y,
                      a.acceleration.z, g.gyro.x, g.gyro.y, g.gyro.z);
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
                int newAngle = random(0, 21);   // Random angle between 0 and 20
                int tempPosition = position + newAngle;
                myServo.write(tempPosition);      // Move to random position
                int ticks = random(900, 4000);    // Wait random from 900 ms -2 seconds
                vTaskDelay(pdMS_TO_TICKS(ticks)); // **Use vTaskDelay instead of delay()**
                myServo.write(position);          // Return to the last known position
            }
        }

        else if (senario == 3) { // Moning gradually from 105° to 45° (1° per minute)
            if (last_position > 45) {
                new_position = last_position - step; // Decrease position by step in degrees
                myServo.write(new_position);         // Move the servo
                last_position = new_position;        // Update last position
                int ticks = 60000;                   // Wait 1 minute (60,000 milliseconds)
                vTaskDelay(pdMS_TO_TICKS(ticks));    // Wait 1 minute (60,000 milliseconds)
            }
        }
    }
}

// Try to initialize servo and mpu6050!
void
setup() {
    Serial.begin(115200);
    myServo.attach(servoPin);   // Attach servo to pin 26
    myServo.write(position);    // Set initial position
    randomSeed(analogRead(A0)); // Seed signals from port for random numbers
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