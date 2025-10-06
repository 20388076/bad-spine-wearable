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
#include "esp_task_wdt.h"      // Watchdog timer to reset if tasks hang but
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
#include "RF9.71W120.h"
Eloquent::ML::Port::RandomForest model;
#else
#include "DT9.71W120.h" //"Best_DecisionTree.h"
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

int scenario = 1; // scenario of expirament 1-5
char axis = 'x';  // Can be 'x', 'y', or 'z'

int choice_num = 0; // Will be set based on scenario and axis

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
    1165; // number of samples per window     <-- Change this value  based on the window size in seconds and sample rate
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
const long MAX_RESULTS = 30;
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
float t1, t2;                                    // Measurements computation time variable
unsigned long start;                             // Start time variable
// float fft_real_acc_x[WINDOW], fft_real_gyro_x[WINDOW]; // FFT real parts
float theta_x, theta_y, theta_z; // tilt angles
// FFT parameters
float fft_real[WINDOW];
float fft_mag[WINDOW];

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

float
computeFeature(int featureId) {
    switch (featureId) {
        case 1: return vector_magnitude(acc_x_data, acc_y_data, acc_z_data, WINDOW);
        case 2: return vector_magnitude(gyro_x_data, gyro_y_data, gyro_z_data, WINDOW);
        case 3: return cubic_prod_median(acc_x_data, acc_y_data, acc_z_data, WINDOW);
        case 4: return cubic_prod_median(gyro_x_data, gyro_y_data, gyro_z_data, WINDOW);
        case 5: return derivative_max(acc_x_data, WINDOW, sampleRate);
        case 6: return derivative_max(acc_y_data, WINDOW, sampleRate);
        case 7: return derivative_max(acc_z_data, WINDOW, sampleRate);
        case 8: return derivative_max(gyro_x_data, WINDOW, sampleRate);
        case 9: return derivative_max(gyro_y_data, WINDOW, sampleRate);
        case 10: return derivative_max(gyro_z_data, WINDOW, sampleRate);
        case 11:
        case 12:
        case 13: {
            float tx, ty, tz;
            compute_gravity_and_thetas(acc_x_data, acc_y_data, acc_z_data, WINDOW, tx, ty, tz);
            if (featureId == 11) {
                return tx;
            }
            if (featureId == 12) {
                return ty;
            }
            return tz;
        }
        case 14: return window_mean(acc_x_data, WINDOW);
        case 15: return window_max(acc_x_data, WINDOW);
        case 16: return window_min(acc_x_data, WINDOW);
        case 17: return window_mean(acc_y_data, WINDOW);
        case 18: return window_max(acc_y_data, WINDOW);
        case 19: return window_min(acc_y_data, WINDOW);
        case 20: return window_mean(acc_z_data, WINDOW);
        case 21: return window_max(acc_z_data, WINDOW);
        case 22: return window_min(acc_z_data, WINDOW);
        case 23: return window_mean(gyro_x_data, WINDOW);
        case 24: return window_max(gyro_x_data, WINDOW);
        case 25: return window_min(gyro_x_data, WINDOW);
        case 26: return window_mean(gyro_y_data, WINDOW);
        case 27: return window_max(gyro_y_data, WINDOW);
        case 28: return window_min(gyro_y_data, WINDOW);
        case 29: return window_mean(gyro_z_data, WINDOW);
        case 30: return window_max(gyro_z_data, WINDOW);
        case 31: return window_min(gyro_z_data, WINDOW);
        case 32: return compute_sma_median(acc_x_data, acc_y_data, acc_z_data, WINDOW);
        case 33: return compute_sma_median(gyro_x_data, gyro_y_data, gyro_z_data, WINDOW);
        case 34: return compute_rms(acc_x_data, WINDOW);
        case 35: return compute_rms(acc_y_data, WINDOW);
        case 36: return compute_rms(acc_z_data, WINDOW);
        case 37: return compute_rms(gyro_x_data, WINDOW);
        case 38: return compute_rms(gyro_y_data, WINDOW);
        case 39: return compute_rms(gyro_z_data, WINDOW);
        case 40: return compute_mad(acc_x_data, WINDOW);
        case 41: return compute_mad(acc_y_data, WINDOW);
        case 42: return compute_mad(acc_z_data, WINDOW);
        case 43: return compute_mad(gyro_x_data, WINDOW);
        case 44: return compute_mad(gyro_y_data, WINDOW);
        case 45: return compute_mad(gyro_z_data, WINDOW);
        case 46: return compute_var(acc_x_data, WINDOW);
        case 47: return compute_var(acc_y_data, WINDOW);
        case 48: return compute_var(acc_z_data, WINDOW);
        case 49: return compute_var(gyro_x_data, WINDOW);
        case 50: return compute_var(gyro_y_data, WINDOW);
        case 51: return compute_var(gyro_z_data, WINDOW);
        case 52: return compute_std(acc_x_data, WINDOW);
        case 53: return compute_std(acc_y_data, WINDOW);
        case 54: return compute_std(acc_z_data, WINDOW);
        case 55: return compute_std(gyro_x_data, WINDOW);
        case 56: return compute_std(gyro_y_data, WINDOW);
        case 57: return compute_std(gyro_z_data, WINDOW);
        case 58: return compute_iqr(acc_x_data, WINDOW);
        case 59: return compute_iqr(acc_y_data, WINDOW);
        case 60: return compute_iqr(acc_z_data, WINDOW);
        case 61: return compute_iqr(gyro_x_data, WINDOW);
        case 62: return compute_iqr(gyro_y_data, WINDOW);
        case 63: return compute_iqr(gyro_z_data, WINDOW);
        case 64: compute_FFT_real(acc_x_data, fft_real, WINDOW); return fft_real[1];
        case 65: compute_FFT_real(acc_y_data, fft_real, WINDOW); return fft_real[1];
        case 66: compute_FFT_real(acc_z_data, fft_real, WINDOW); return fft_real[1];
        case 67: compute_FFT_real(gyro_x_data, fft_real, WINDOW); return fft_real[1];
        case 68: compute_FFT_real(gyro_y_data, fft_real, WINDOW); return fft_real[1];
        case 69: compute_FFT_real(gyro_z_data, fft_real, WINDOW); return fft_real[1];
        case 70: compute_FFT_mag(acc_x_data, fft_mag, WINDOW); return compute_fft_energy(fft_mag, WINDOW);
        case 71: compute_FFT_mag(acc_y_data, fft_mag, WINDOW); return compute_fft_energy(fft_mag, WINDOW);
        case 72: compute_FFT_mag(acc_z_data, fft_mag, WINDOW); return compute_fft_energy(fft_mag, WINDOW);
        case 73: compute_FFT_mag(gyro_x_data, fft_mag, WINDOW); return compute_fft_energy(fft_mag, WINDOW);
        case 74: compute_FFT_mag(gyro_y_data, fft_mag, WINDOW); return compute_fft_energy(fft_mag, WINDOW);
        case 75: compute_FFT_mag(gyro_z_data, fft_mag, WINDOW); return compute_fft_energy(fft_mag, WINDOW);
        default: return 0.0f;
    }
}

// Task 1: Read MPU6050 data, compute features, and make predictions

void
Task1code(void* pvParameters) {
    long iteration = 0;
    int correct = 0;
    int total = MAX_RESULTS;
    while (iteration < MAX_RESULTS) {
        sample_index = 0;

        for (int i = 0; i < WINDOW; i++) {
            start = millis();
            sensors_event_t a, g, temp;
            mpu.getEvent(&a, &g, &temp);
            acc_x_data[i] = a.acceleration.x / G_CONST; // Convert acceleration from m/s^2 to g
            acc_y_data[i] = a.acceleration.y / G_CONST;
            acc_z_data[i] = a.acceleration.z / G_CONST;
            gyro_x_data[i] = g.gyro.x;
            gyro_y_data[i] = g.gyro.y;
            gyro_z_data[i] = g.gyro.z;

            sample_index = i;

            if (sample_index < WINDOW - 1) {
                t1 = millis() - start;
                //Serial.printf("\nComputation time1: %.2f ms\n", t1);
                vTaskDelay(pdMS_TO_TICKS(
                    samplePeriod
                    - t1)); // If e.g sample is at 50 Hz (every 1000/50 = 20 ms - processing time) wait to achieve 50 Hz

            } else if (sample_index == WINDOW - 1) {
                // float output_matrix[75]; // Initialize output matrix
                // --- Start feature computations ---
                UBaseType_t freeStack = uxTaskGetStackHighWaterMark(NULL);
                Serial.printf("Task1 stack remaining: %u words (%u bytes)\n", freeStack, freeStack * 4);

                int selectedFeatures[] = {64, 2, 3, 4, 5, 6, 7, 8, 9, 10};
                int numFeatures = sizeof(selectedFeatures) / sizeof(selectedFeatures[0]);
                float values[numFeatures];

                for (int i = 0; i < numFeatures; i++) {
                    values[i] = computeFeature(selectedFeatures[i]);
                }

                int predicted = model.predict(values);

                Serial.printf("\nIteration %ld/%ld - Prediction result: %d\n", iteration, MAX_RESULTS, predicted);
                int actual = (int)y_test[iteration];

                if (predicted == actual) {
                    correct++;
                }

                t2 = millis() - start;
                Serial.printf("\nComputation time2: %.2f ms\n", t2);
                iteration++;
                taskYIELD(); // let idle run
                vTaskDelay(pdMS_TO_TICKS(1));
                vTaskDelay(pdMS_TO_TICKS(
                    (5 * samplePeriod)
                    - t2)); // If e.g sample is at 50 Hz (every 1000/50 = 20 ms - processing time) wait to achieve 50 Hz
            }
        }
    }
    float accuracy = (float)correct / total * 100.0;
    Serial.print("\nTotal Accuracy: ");
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

        else if (scenario == 3 || scenario == 4
                 || scenario == 5) { // Moving gradually from 85° to 45° with servo steps per minute
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
    setCpuFrequencyMhz(240);  // Sets the CPU frequency to 240 MHz
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
                            20000,     /* Stack size of task */
                            NULL,      /* parameter of the task */
                            1,         /* priority of the task */
                            &Task1,    /* Task handle to keep track of created task */
                            0);        /* pin task to core 0 */
    delay(100);

    // Create a task that will be executed in the Task2code() function, with priority 1 and executed on core 1
    xTaskCreatePinnedToCore(Task2code, /* Task function. */
                            "Task2",   /* name of task. */
                            20000,     /* Stack size of task */
                            NULL,      /* parameter of the task */
                            1,         /* priority of the task */
                            &Task2,    /* Task handle to keep track of created task */
                            1);        /* pin task to core 1 */
    // Disable watchdog on the current task (core 0 Task1)
    disableCore0WDT();
    //disableCore1WDT();
    esp_task_wdt_deinit(); // fully stop WDT globally (debug only!)
    delay(100);
}

void
loop() {}
