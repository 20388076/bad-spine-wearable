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

/* Include the Classification model file */

#define USE_RAW_DATA 0 // Set to 0 for DecisionTree, 1 for RandomForest RELIEF features

#if USE_RAW_DATA
#include "Best_RandomForest.h"
Eloquent::ML::Port::RandomForest model;
#else
#include "Best_DecisionTree.h"
Eloquent::ML::Port::DecisionTree model;
#endif

/* User configuration */

// Senario of expirament
// 1: No movement detection (for x - > choice_num = 0, y - > choice_num = 1, z - > choice_num = 2)
// 2: Random movement detection (for x - > choice_num = 3, y - > choice_num = 4, z - > choice_num = 5)
// 3: Gradual movement detection 1 step per minute (for x - > choice_num = 6, y - > choice_num = 7, z - > choice_num = 8)
// 4: Gradual movement detection 2 steps per minute (for x - > choice_num = 9, y - > choice_num = 10, z - > choice_num = 11)
// 5: Gradual movement detection 3 steps per minute with random movement / anomaly detection (for x - > choice_num = 12, y - > choice_num = 13, z - > choice_num = 14)
// Choice number for test data (0-14 for different test scenarios)
int choice_num = 0; // Will be set based on scenario and axis
int scenario = 1;   // scenario of expirament 1-5
char axis = 'z';    // Can be 'x', 'y', or 'z'

// Function to set choice_num based on scenario and axis
void
setChoiceNum() {
    int axisOffset;

    // Convert axis character to offset (x=0, y=1, z=2)
    switch (axis) {
        case 'x': axisOffset = 0; break;
        case 'y': axisOffset = 1; break;
        case 'z': axisOffset = 2; break;
        default: axisOffset = 0; // default to x axis
    }

    // Calculate choice_num based on scenario and axis
    switch (scenario) {
        case 1:                          // No movement detection
            choice_num = 0 + axisOffset; // x->0, y->1, z->2
            break;
        case 2:                          // Random movement detection
            choice_num = 3 + axisOffset; // x->3, y->4, z->5
            break;
        case 3:                          // Gradual movement 1 step/min
            choice_num = 6 + axisOffset; // x->6, y->7, z->8
            break;
        case 4:                          // Gradual movement 2 steps/min
            choice_num = 9 + axisOffset; // x->9, y->10, z->11
            break;
        case 5:                           // Gradual movement 3 steps/min with anomaly
            choice_num = 12 + axisOffset; // x->12, y->13, z->14
            break;
        default: choice_num = 0; // Default to x-axis, no movement
    }

    Serial.printf("Scenario %d, Axis %c -> choice_num = %d\n", scenario, axis, choice_num);
}

// Sampling configuration
float sampleRate = 9.71f; // Sample rate in Hz       <-- Change this value  according to your needs

const int WINDOW =
    19; // number of samples per window     <-- Change this value  based on the window size in seconds and sample rate
// WINDOW = round( sampleRate * window size in seconds)
// e.g. for 2 second window and sample rate 9.71 Hz, WINDOW = 19
// e.g. for 2 second window and sample rate 10 Hz, WINDOW = 20
// e.g. for 2 second window and sample rate 50 Hz, WINDOW = 100

/* Servo Variables configuration */

int initial_position = 85;      // Start at 85 degrees
int min_position = 45;          // Minimum position of servo motor
static const int servoPin = 25; // Connect servo to pin 25 or D2
unsigned long previousMillis = 0;
int new_position = 0;                 // New position for servo motor
int last_position = initial_position; // Last position of servo motor
int step;                             // Factor to decrease position by degrees per miniute

/* Sampling and Features Variables configuration */
const long MAX_RESULTS = 1846;
float y_test[MAX_RESULTS]; // Array to hold test values

// Initialize all elements with choice_num
void
initializeTestArray() {
    for (int i = 0; i < MAX_RESULTS; i++) {
        y_test[i] = choice_num;
    }
}

float acc_x_data[WINDOW], acc_y_data[WINDOW], gyro_y_data[WINDOW], gyro_z_data[WINDOW], gyro_x_data[WINDOW],
    acc_z_data[WINDOW];                          // data arrays
int sample_index = 0;                            // current sample index in window
const float G_CONST = 9.80665f;                  // Standard gravity
float samplePeriod = round(1000.0 / sampleRate); // Sample period in ms
float t;                                         // Measurements computation time variable
unsigned long start;                             // Start time variable
// float fft_real_acc_x[WINDOW], fft_real_gyro_x[WINDOW]; // FFT real parts
float theta_x, theta_y, theta_z; // tilt angles
// FFT parameters
float fft_real_acc_x[WINDOW];
float fft_mag_acc_x[WINDOW];
float fft_real_acc_y[WINDOW];
float fft_mag_acc_y[WINDOW];
float fft_real_acc_z[WINDOW];
float fft_mag_acc_z[WINDOW];
float fft_real_gyro_x[WINDOW];
float fft_mag_gyro_x[WINDOW];
float fft_real_gyro_y[WINDOW];
float fft_mag_gyro_y[WINDOW];
float fft_real_gyro_z[WINDOW];
float fft_mag_gyro_z[WINDOW];

/* Features Functions */

// ---- Mean ----
float
window_mean(float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / n;
}

// ---- Max ----
float
window_max(float* data, int n) {
    float m = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] > m) {
            m = data[i];
        }
    }
    return m;
}

// ---- Min ----
float
window_min(float* data, int n) {
    float m = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] < m) {
            m = data[i];
        }
    }
    return m;
}

// ---- RMS ----
float
compute_rms(float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i] * data[i];
    }
    return sqrt(sum / n);
}

// ---- Variance ----
float
compute_var(float* data, int n) {
    float m = window_mean(data, n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = data[i] - m;
        sum += d * d;
    }
    return sum / n;
}

// ---- Std ----
float
compute_std(float* data, int n) {
    return sqrt(compute_var(data, n));
}

// ---- MAD ----
float
compute_mad(float* data, int n) {
    float m = window_mean(data, n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += fabs(data[i] - m);
    }
    return sum / n;
}

// ---- IQR ----
float
compute_iqr(float* data, int len) {
    float sorted[len];
    memcpy(sorted, data, sizeof(float) * len);
    std::sort(sorted, sorted + len);

    float q1_pos = (len - 1) * 0.25;
    float q3_pos = (len - 1) * 0.75;

    int q1_idx = (int)q1_pos;
    int q3_idx = (int)q3_pos;

    float q1 = sorted[q1_idx] + (sorted[q1_idx + 1] - sorted[q1_idx]) * (q1_pos - q1_idx);
    float q3 = sorted[q3_idx] + (sorted[q3_idx + 1] - sorted[q3_idx]) * (q3_pos - q3_idx);

    return q3 - q1;
}

// ---- FFT ----
void
compute_FFT_real(float* input, float* real_out, int size) {
    FFT_real myFFT(size);
    myFFT.setInput(input);
    myFFT.compute(); // Calculate real and imaginary parts

    float* real = myFFT.getReal();

    for (int i = 0; i < size; ++i) {
        real_out[i] = real[i];
    }
}

// ---- FFT Energy ----
// Compute only magnitude
void
compute_FFT_mag(float* input, float* magnitude_out, int size) {
    FFT_real myFFT(size);
    myFFT.setInput(input);
    myFFT.compute();          // Calculate real and imaginary parts
    myFFT.computeMagnitude(); // Compute magnitude

    float* mag = myFFT.getMagnitude();

    for (int i = 0; i < size; ++i) {
        magnitude_out[i] = mag[i];
    }
}

float
compute_fft_energy(float* magnitude, int len) {
    float sum = 0;

    for (int i = 0; i < len; i++) {
        sum += magnitude[i] * magnitude[i];
    }

    return sum / len;
}

// ---- SMA median ----
float
compute_sma_median(float* ax, float* ay, float* az, int n) {
    float vals[n];
    for (int i = 0; i < n; i++) {
        vals[i] = fabs(ax[i]) + fabs(ay[i]) + fabs(az[i]);
    }
    // sort
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (vals[j] > vals[j + 1]) {
                float tmp = vals[j];
                vals[j] = vals[j + 1];
                vals[j + 1] = tmp;
            }
        }
    }
    return vals[n / 2];
}

// ---- Median ----
float
compute_median(float* data, int n) {
    float copy[n];
    for (int i = 0; i < n; i++) {
        copy[i] = data[i];
    }

    // sort copy[] ascending (simple bubble sort for small n=16)
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (copy[j] > copy[j + 1]) {
                float tmp = copy[j];
                copy[j] = copy[j + 1];
                copy[j + 1] = tmp;
            }
        }
    }

    if (n % 2 == 0) {
        // even count → average middle two
        return (copy[n / 2 - 1] + copy[n / 2]) / 2.0f;
    } else {
        // odd count → take middle
        return copy[n / 2];
    }
}

// ---- Vector Magnitude ----
float
vector_magnitude(float* x, float* y, float* z, int n) {
    float s[n];
    for (int i = 0; i < n; i++) {
        // per-sample Euclidean magnitude
        s[i] = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
    }
    // return the mean magnitude over the window
    return compute_median(s, n);
}

// ---- Cubic Product Magnitude ----
float
cubic_prod_median(float* x, float* y, float* z, int n) {
    float sum = 0.0f;
    float power[n];
    for (int i = 0; i < n; i++) {
        float prod = fabs(x[i] * y[i] * z[i]);
        power[i] = pow(prod, 1.0f / 3.0f);
    }
    return compute_median(power, n);
}

// ---- Derivative max ----
float
derivative_max(float* data, int n, float sampleRate) {
    float maxv = fabs((data[1] - data[0]) / sampleRate);
    for (int i = 1; i < n; i++) {
        float v = fabs((data[i] - data[i - 1]) / sampleRate);
        if (v > maxv) {
            maxv = v;
        }
    }
    return maxv;
}

// ---- Gravity vector + tilt angles (theta) ----
void
compute_gravity_and_thetas(float* ax_g, float* ay_g, float* az_g, int n, float& theta_x, float& theta_y,
                           float& theta_z) {
    float g_x, g_y, g_z, g_mag;
    // ---- Step 1: Mean per axis (already in g-units)
    g_x = window_mean(ax_g, n);
    g_y = window_mean(ay_g, n);
    g_z = window_mean(az_g, n);

    // ---- Step 2: Gravity magnitude
    g_mag = sqrt(g_x * g_x + g_y * g_y + g_z * g_z);

    // ---- Step 3: Compute tilt angles in degrees
    float cx = g_x / g_mag;
    float cy = g_y / g_mag;
    float cz = g_z / g_mag;
    theta_x = acos(cx);
    theta_y = acos(cy);
    theta_z = acos(cz);
}

// Task 1: Read MPU6050 data, compute features, and make predictions
float t1, t2;

void
Task1code(void* pvParameters) {
    long iteration = 0;
    int correct = 0;
    int total = MAX_RESULTS;
    while (iteration < MAX_RESULTS) {
        start = millis();
        sample_index = 0;

        for (int i = 0; i < WINDOW; i++) {
            sensors_event_t a, g, temp;
            mpu.getEvent(&a, &g, &temp);
            acc_x_data[i] = a.acceleration.x / G_CONST; // Convert acceleration from m/s^2 to g
            acc_y_data[i] = a.acceleration.y / G_CONST;
            acc_z_data[i] = a.acceleration.z / G_CONST;
            gyro_x_data[i] = g.gyro.x;
            gyro_y_data[i] = g.gyro.y;
            gyro_z_data[i] = g.gyro.z;
            sample_index = i;
            vTaskDelay(pdMS_TO_TICKS(samplePeriod));
        }

        t1 = millis() - start;
        Serial.printf("\nComputation time: %.2f ms\n", t1);

        start = millis();
        if (sample_index == WINDOW - 1) {

            // float output_matrix[75]; // Initialize output matrix

            // --- f1 ---
            float f1 = vector_magnitude(acc_x_data, acc_y_data, acc_z_data, WINDOW);
            // --- f2 ---
            float f2 = vector_magnitude(gyro_x_data, gyro_y_data, gyro_z_data, WINDOW);
            // --- f3 ---
            float f3 = cubic_prod_median(acc_x_data, acc_y_data, acc_z_data, WINDOW);
            // --- f4 ---
            float f4 = cubic_prod_median(gyro_x_data, gyro_y_data, gyro_z_data, WINDOW);
            // --- f5 ---
            float f5 = derivative_max(acc_x_data, WINDOW, sampleRate);
            // --- f6 ---
            float f6 = derivative_max(acc_y_data, WINDOW, sampleRate);
            // --- f7 ---
            float f7 = derivative_max(acc_z_data, WINDOW, sampleRate);
            // --- f8 ---
            float f8 = derivative_max(gyro_x_data, WINDOW, sampleRate);
            // --- f9 ---
            float f9 = derivative_max(gyro_y_data, WINDOW, sampleRate);
            // --- f10 ---
            float f10 = derivative_max(gyro_z_data, WINDOW, sampleRate);
            // --- f11–f13 (thetas already computed) ---
            compute_gravity_and_thetas(acc_x_data, acc_y_data, acc_z_data, WINDOW, theta_x, theta_y, theta_z);
            float f11 = theta_x;
            float f12 = theta_y;
            float f13 = theta_z;
            // --- f14 ---
            float f14 = window_mean(acc_x_data, WINDOW);
            // --- f15 ---
            float f15 = window_max(acc_x_data, WINDOW);
            // --- f16 ---
            float f16 = window_min(acc_x_data, WINDOW);
            // --- f17 ---
            float f17 = window_mean(acc_y_data, WINDOW);
            // --- f18 ---
            float f18 = window_max(acc_y_data, WINDOW);
            // --- f19 ---
            float f19 = window_min(acc_y_data, WINDOW);
            // --- f20 ---
            float f20 = window_mean(acc_z_data, WINDOW);
            // --- f21 ---
            float f21 = window_max(acc_z_data, WINDOW);
            // --- f22 ---
            float f22 = window_min(acc_z_data, WINDOW);
            // --- f23 ---
            float f23 = window_mean(gyro_x_data, WINDOW);
            // --- f24 ---
            float f24 = window_max(gyro_x_data, WINDOW);
            // --- f25 ---
            float f25 = window_min(gyro_x_data, WINDOW);
            // --- f26 ---
            float f26 = window_mean(gyro_y_data, WINDOW);
            // --- f27 ---
            float f27 = window_max(gyro_y_data, WINDOW);
            // --- f28 ---
            float f28 = window_min(gyro_y_data, WINDOW);
            // --- f29 ---
            float f29 = window_mean(gyro_z_data, WINDOW);
            // --- f30 ---
            float f30 = window_max(gyro_z_data, WINDOW);
            // --- f31 ---
            float f31 = window_min(gyro_z_data, WINDOW);
            // --- f32 ---
            float f32 = compute_sma_median(acc_x_data, acc_y_data, acc_z_data, WINDOW);
            // --- f33 ---
            float f33 = compute_sma_median(gyro_x_data, gyro_y_data, gyro_z_data, WINDOW);
            // --- f34 ---
            float f34 = compute_rms(acc_x_data, WINDOW);
            // --- f35 ---
            float f35 = compute_rms(acc_y_data, WINDOW);
            // --- f36 ---
            float f36 = compute_rms(acc_z_data, WINDOW);
            // --- f37 ---
            float f37 = compute_rms(gyro_x_data, WINDOW);
            // --- f38 ---
            float f38 = compute_rms(gyro_y_data, WINDOW);
            // --- f39 ---
            float f39 = compute_rms(gyro_z_data, WINDOW);
            // --- f40 ---
            float f40 = compute_mad(acc_x_data, WINDOW);
            // --- f41 ---
            float f41 = compute_mad(acc_y_data, WINDOW);
            // --- f42 ---
            float f42 = compute_mad(acc_z_data, WINDOW);
            // --- f43 ---
            float f43 = compute_mad(gyro_x_data, WINDOW);
            // --- f44 ---
            float f44 = compute_mad(gyro_y_data, WINDOW);
            // --- f45 ---
            float f45 = compute_mad(gyro_z_data, WINDOW);
            // --- f46 ---
            float f46 = compute_var(acc_x_data, WINDOW);
            // --- f47 ---
            float f47 = compute_var(acc_y_data, WINDOW);
            // --- f48 ---
            float f48 = compute_var(acc_z_data, WINDOW);
            // --- f49 ---
            float f49 = compute_var(gyro_x_data, WINDOW);
            // --- f50 ---
            float f50 = compute_var(gyro_y_data, WINDOW);
            // --- f51 ---
            float f51 = compute_var(gyro_z_data, WINDOW);
            float f52 = compute_std(acc_x_data, WINDOW);
            float f53 = compute_std(acc_y_data, WINDOW);
            float f54 = compute_std(acc_z_data, WINDOW);
            float f55 = compute_std(gyro_x_data, WINDOW);
            float f56 = compute_std(gyro_y_data, WINDOW);
            float f57 = compute_std(gyro_z_data, WINDOW);
            float f58 = compute_iqr(acc_x_data, WINDOW);
            float f59 = compute_iqr(acc_y_data, WINDOW);
            float f60 = compute_iqr(acc_z_data, WINDOW);
            float f61 = compute_iqr(gyro_x_data, WINDOW);
            float f62 = compute_iqr(gyro_y_data, WINDOW);
            float f63 = compute_iqr(gyro_z_data, WINDOW);
            compute_FFT_real(acc_x_data, fft_real_acc_x, WINDOW);
            float f64 = fft_real_acc_x[1]; // first non-DC bin
            compute_FFT_real(acc_y_data, fft_real_acc_y, WINDOW);
            float f65 = fft_real_acc_y[1];
            compute_FFT_real(acc_z_data, fft_real_acc_z, WINDOW);
            float f66 = fft_real_acc_z[1]; //  float f66 = fft_real_acc_z[8]; // first non-DC bin
            compute_FFT_real(gyro_x_data, fft_real_gyro_x, WINDOW);
            float f67 = fft_real_gyro_x[1];
            compute_FFT_real(gyro_y_data, fft_real_gyro_y, WINDOW);
            float f68 = fft_real_gyro_y[1];
            compute_FFT_real(gyro_z_data, fft_real_gyro_z, WINDOW);
            float f69 = fft_real_gyro_z[1];
            compute_FFT_mag(acc_x_data, fft_mag_acc_x, WINDOW);
            float f70 = compute_fft_energy(fft_mag_acc_x, WINDOW);
            compute_FFT_mag(acc_y_data, fft_mag_acc_y, WINDOW);
            float f71 = compute_fft_energy(fft_mag_acc_y, WINDOW);
            compute_FFT_mag(acc_z_data, fft_mag_acc_z, WINDOW);
            float f72 = compute_fft_energy(fft_mag_acc_z, WINDOW);
            compute_FFT_mag(gyro_x_data, fft_mag_gyro_x, WINDOW);
            float f73 = compute_fft_energy(fft_mag_gyro_x, WINDOW);
            compute_FFT_mag(gyro_y_data, fft_mag_gyro_y, WINDOW);
            float f74 = compute_fft_energy(fft_mag_gyro_y, WINDOW);
            compute_FFT_mag(gyro_z_data, fft_mag_gyro_z, WINDOW);
            float f75 = compute_fft_energy(fft_mag_gyro_z, WINDOW);

            float values[] = {f1,  f2,  f3,  f4,  f5,  f6,  f7,  f8,  f9,  f10, f11, f12, f13, f14, f15,
                              f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
                              f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
                              f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
                              f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75};

            int predicted = model.predict(values);

            Serial.printf("\nIteration %ld/%ld - Prediction result: %d\n", iteration, MAX_RESULTS, predicted);
            int actual = (int)y_test[iteration];

            if (predicted == actual) {
                correct++;
            }
        }
        t = millis() - start;
        Serial.printf("\nComputation time: %.2f ms\n", t);
        iteration++;
        //vTaskDelay(pdMS_TO_TICKS(samplePeriod - t)); // Sample is at 50 Hz (every 1000/50 = 20 ms - processing time) wait to achieve 50 Hz
    }
    float accuracy = (float)correct / total * 100.0;
    Serial.print("Total Accuracy: ");
    Serial.print(accuracy);
    Serial.println("%");
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
    Serial.begin(115200);     // Start serial communication at 115200 baud rate
    setChoiceNum();           // Set choice_num based on scenario and axis
    initializeTestArray();    // Initialize y_test array with choice_num
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
