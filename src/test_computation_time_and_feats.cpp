#include <Arduino.h>
#include <cmath>   // sqrt, acos
#include <algorithm> // std::max, std::min
#include "fft.h" // your custom FFT implementation

// ===========================================
// Constants
// ===========================================
const float G_CONST = 9.80665f;
const int WINDOW = 16; // number of samples per window
const float DT = 103;  // 100 Hz sampling -> 10 ms
float times[75];
float theta_x, theta_y, theta_z;
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

// ===========================================
// Test Data (from add.txt)
// Replace these with your full arrays if truncated
// ===========================================
float acc_x_data[WINDOW] = {0.192, 0.172, 0.196, 0.182, 0.196, 0.192, 0.165, 0.180,
                            0.172, 0.180, 0.194, 0.187, 0.196, 0.208, 0.182, 0.199};

float acc_y_data[WINDOW] = {0.048, 0.055, 0.043, 0.036, 0.022, 0.045, 0.060, 0.045,
                            0.034, 0.055, 0.048, 0.045, 0.043, 0.038, 0.038, 0.055};

float acc_z_data[WINDOW] = {10.362, 10.372, 10.353, 10.372, 10.357, 10.381, 10.364, 10.403,
                            10.367, 10.364, 10.405, 10.376, 10.400, 10.369, 10.367, 10.362};

float gyro_x_data[WINDOW] = {-0.009, -0.009, -0.007, -0.009, -0.009, -0.009, -0.009, -0.009,
                             -0.009, -0.009, -0.008, -0.009, -0.009, -0.009, -0.007, -0.008};

float gyro_y_data[WINDOW] = {0.015, 0.014, 0.015, 0.015, 0.016, 0.015, 0.014, 0.015,
                             0.015, 0.015, 0.014, 0.013, 0.014, 0.014, 0.014, 0.015};

float gyro_z_data[WINDOW] = {-0.016, -0.017, -0.017, -0.016, -0.017, -0.016, -0.017, -0.017,
                             -0.017, -0.015, -0.015, -0.017, -0.015, -0.016, -0.016, -0.015};

// ===========================================
// Features Functions
// ===========================================

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
derivative_max(float* data, int n, float dt) {
    float maxv = fabs((data[1] - data[0]) / dt);
    for (int i = 1; i < n; i++) {
        float v = fabs((data[i] - data[i - 1]) / dt);
        if (v > maxv) {
            maxv = v;
        }
    }
    return maxv;
}

// ---- Thetas (median) ----
void
compute_thetas(float* ax, float* ay, float* az, int n, float& theta_x, float& theta_y, float& theta_z) {
    float th_x_arr[n], th_y_arr[n], th_z_arr[n];

    for (int i = 0; i < n; i++) {
        float g_mag = std::sqrt(ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i]);
        if (g_mag == 0) {
            th_x_arr[i] = th_y_arr[i] = th_z_arr[i] = 0.0f;
            continue;
        }

        float cx = ax[i] / g_mag;
        float cy = ay[i] / g_mag;
        float cz = az[i] / g_mag;

        // clamp for safety
        cx = std::max(-1.0f, std::min(1.0f, cx));
        cy = std::max(-1.0f, std::min(1.0f, cy));
        cz = std::max(-1.0f, std::min(1.0f, cz));

        th_x_arr[i] = std::acos(cx);
        th_y_arr[i] = std::acos(cy);
        th_z_arr[i] = std::acos(cz);
    }

    // Return median per axis
    theta_x = compute_median(th_x_arr, n);
    theta_y = compute_median(th_y_arr, n);
    theta_z = compute_median(th_z_arr, n);
}

void
printCSV(const float* arr, int size) {
    for (int i = 0; i < size; i++) {
        Serial.print(arr[i], 3);
        if (i < size - 1) {
            Serial.print(",");
        }
    }
}

// ===========================================
// ESP32 setup
// ===========================================
void
setup() {
    Serial.begin(115200);
    delay(1000);

    // Convert accel to g
    for (int i = 0; i < WINDOW; i++) {
        acc_x_data[i] /= G_CONST;
        acc_y_data[i] /= G_CONST;
        acc_z_data[i] /= G_CONST;
    }

    // Print header
    Serial.println(
        "SVM_a,SVM_g,CM_a,CM_g,jerk_x,jerk_y,jerk_z,accl_x,accl_y,accl_z,th_x,th_y,"
        "th_z,ag_x_mean,ag_x_max,ag_x_min,ag_y_mean,ag_y_max,ag_y_min,ag_z_mean,ag_z_max,ag_z_min,g_x_mean,g_x_max,"
        "g_x_min,g_y_mean,g_y_max,g_y_min,g_z_mean,g_z_max,g_z_min,SMA_a,SMA_g,RMS_ag_x,RMS_ag_y,RMS_ag_z,"
        "RMS_g_x,RMS_g_y,RMS_g_z,MAD_ag_x,MAD_ag_y,MAD_ag_z,MAD_g_x,MAD_g_y,MAD_g_z,VAR_ag_x,VAR_ag_y,VAR_ag_z,"
        "VAR_g_x,VAR_g_y,VAR_g_z,STD_ag_x,STD_ag_y,STD_ag_z,STD_g_x,STD_g_y,STD_g_z,IQR_ag_x,IQR_ag_y,IQR_ag_z,"
        "IQR_g_x,IQR_g_y,IQR_g_z,FFT_ag_x,FFT_ag_y,FFT_ag_z,FFT_g_x,FFT_g_y,FFT_g_z,E_ag_x,E_ag_y,E_ag_z,E_g_x,E_g_y,E_"
        "g_z");

    unsigned long start;

    // --- f1 ---
    start = micros();
    float f1 = vector_magnitude(acc_x_data, acc_y_data, acc_z_data, WINDOW);
    float t1 = micros() - start; // timw in microseconds

    // --- f2 ---
    start = micros();
    float f2 = vector_magnitude(gyro_x_data, gyro_y_data, gyro_z_data, WINDOW);
    float t2 = micros() - start;

    // --- f3 ---
    start = micros();
    float f3 = cubic_prod_median(acc_x_data, acc_y_data, acc_z_data, WINDOW);
    float t3 = micros() - start;

    // --- f4 ---
    start = micros();
    float f4 = cubic_prod_median(gyro_x_data, gyro_y_data, gyro_z_data, WINDOW);
    float t4 = micros() - start;

    // --- f5 ---
    start = micros();
    float f5 = derivative_max(acc_x_data, WINDOW, DT);
    float t5 = micros() - start;

    // --- f6 ---
    start = micros();
    float f6 = derivative_max(acc_y_data, WINDOW, DT);
    float t6 = micros() - start;

    // --- f7 ---
    start = micros();
    float f7 = derivative_max(acc_z_data, WINDOW, DT);
    float t7 = micros() - start;

    // --- f8 ---
    start = micros();
    float f8 = derivative_max(gyro_x_data, WINDOW, DT);
    float t8 = micros() - start;

    // --- f9 ---
    start = micros();
    float f9 = derivative_max(gyro_y_data, WINDOW, DT);
    float t9 = micros() - start;

    // --- f10 ---
    start = micros();
    float f10 = derivative_max(gyro_z_data, WINDOW, DT);
    float t10 = micros() - start;

    // --- f11–f13 (thetas already computed) ---
    start = micros();
    compute_thetas(acc_x_data, acc_y_data, acc_z_data, WINDOW, theta_x, theta_y, theta_z);
    float elapsedTheta = micros() - start;
    float f11 = theta_x;
    float f12 = theta_y;
    float f13 = theta_z;
    float t11 = elapsedTheta / 3; // average time per theta
    float t12 = elapsedTheta / 3;
    float t13 = elapsedTheta / 3;

    // --- f14 ---
    start = micros();
    float f14 = window_mean(acc_x_data, WINDOW);
    float t14 = micros() - start;

    // --- f15 ---
    start = micros();
    float f15 = window_max(acc_x_data, WINDOW);
    float t15 = micros() - start;

    // --- f16 ---
    start = micros();
    float f16 = window_min(acc_x_data, WINDOW);
    float t16 = micros() - start;

    // --- f17 ---
    start = micros();
    float f17 = window_mean(acc_y_data, WINDOW);
    float t17 = micros() - start;

    // --- f18 ---
    start = micros();
    float f18 = window_max(acc_y_data, WINDOW);
    float t18 = micros() - start;

    // --- f19 ---
    start = micros();
    float f19 = window_min(acc_y_data, WINDOW);
    float t19 = micros() - start;

    // --- f20 ---
    start = micros();
    float f20 = window_mean(acc_z_data, WINDOW);
    float t20 = micros() - start;

    // --- f21 ---
    start = micros();
    float f21 = window_max(acc_z_data, WINDOW);
    float t21 = micros() - start;

    // --- f22 ---
    start = micros();
    float f22 = window_min(acc_z_data, WINDOW);
    float t22 = micros() - start;

    // --- f23 ---
    start = micros();
    float f23 = window_mean(gyro_x_data, WINDOW);
    float t23 = micros() - start;

    // --- f24 ---
    start = micros();
    float f24 = window_max(gyro_x_data, WINDOW);
    float t24 = micros() - start;

    // --- f25 ---
    start = micros();
    float f25 = window_min(gyro_x_data, WINDOW);
    float t25 = micros() - start;

    // --- f26 ---
    start = micros();
    float f26 = window_mean(gyro_y_data, WINDOW);
    float t26 = micros() - start;

    // --- f27 ---
    start = micros();
    float f27 = window_max(gyro_y_data, WINDOW);
    float t27 = micros() - start;

    // --- f28 ---
    start = micros();
    float f28 = window_min(gyro_y_data, WINDOW);
    float t28 = micros() - start;

    // --- f29 ---
    start = micros();
    float f29 = window_mean(gyro_z_data, WINDOW);
    float t29 = micros() - start;

    // --- f30 ---
    start = micros();
    float f30 = window_max(gyro_z_data, WINDOW);
    float t30 = micros() - start;

    // --- f31 ---
    start = micros();
    float f31 = window_min(gyro_z_data, WINDOW);
    float t31 = micros() - start;

    // --- f32 ---
    start = micros();
    float f32 = compute_sma_median(acc_x_data, acc_y_data, acc_z_data, WINDOW);
    float t32 = micros() - start;

    // --- f33 ---
    start = micros();
    float f33 = compute_sma_median(gyro_x_data, gyro_y_data, gyro_z_data, WINDOW);
    float t33 = micros() - start;

    // --- f34 ---
    start = micros();
    float f34 = compute_rms(acc_x_data, WINDOW);
    float t34 = micros() - start;

    // --- f35 ---
    start = micros();
    float f35 = compute_rms(acc_y_data, WINDOW);
    float t35 = micros() - start;

    // --- f36 ---
    start = micros();
    float f36 = compute_rms(acc_z_data, WINDOW);
    float t36 = micros() - start;

    // --- f37 ---
    start = micros();
    float f37 = compute_rms(gyro_x_data, WINDOW);
    float t37 = micros() - start;

    // --- f38 ---
    start = micros();
    float f38 = compute_rms(gyro_y_data, WINDOW);
    float t38 = micros() - start;

    // --- f39 ---
    start = micros();
    float f39 = compute_rms(gyro_z_data, WINDOW);
    float t39 = micros() - start;

    // --- f40 ---
    start = micros();
    float f40 = compute_mad(acc_x_data, WINDOW);
    float t40 = micros() - start;

    // --- f41 ---
    start = micros();
    float f41 = compute_mad(acc_y_data, WINDOW);
    float t41 = micros() - start;

    // --- f42 ---
    start = micros();
    float f42 = compute_mad(acc_z_data, WINDOW);
    float t42 = micros() - start;

    // --- f43 ---
    start = micros();
    float f43 = compute_mad(gyro_x_data, WINDOW);
    float t43 = micros() - start;

    // --- f44 ---
    start = micros();
    float f44 = compute_mad(gyro_y_data, WINDOW);
    float t44 = micros() - start;

    // --- f45 ---
    start = micros();
    float f45 = compute_mad(gyro_z_data, WINDOW);
    float t45 = micros() - start;

    // --- f46 ---
    start = micros();
    float f46 = compute_var(acc_x_data, WINDOW);
    float t46 = micros() - start;

    // --- f47 ---
    start = micros();
    float f47 = compute_var(acc_y_data, WINDOW);
    float t47 = micros() - start;

    // --- f48 ---
    start = micros();
    float f48 = compute_var(acc_z_data, WINDOW);
    float t48 = micros() - start;

    // --- f49 ---
    start = micros();
    float f49 = compute_var(gyro_x_data, WINDOW);
    float t49 = micros() - start;

    // --- f50 ---
    start = micros();
    float f50 = compute_var(gyro_y_data, WINDOW);
    float t50 = micros() - start;

    // --- f51 ---
    start = micros();
    float f51 = compute_var(gyro_z_data, WINDOW);
    float t51 = micros() - start;

    // --- f52 ---
    start = micros();
    float f52 = compute_std(acc_x_data, WINDOW);
    float t52 = micros() - start;

    // --- f53 ---
    start = micros();
    float f53 = compute_std(acc_y_data, WINDOW);
    float t53 = micros() - start;

    // --- f54 ---
    start = micros();
    float f54 = compute_std(acc_z_data, WINDOW);
    float t54 = micros() - start;

    // --- f55 ---
    start = micros();
    float f55 = compute_std(gyro_x_data, WINDOW);
    float t55 = micros() - start;

    // --- f56 ---
    start = micros();
    float f56 = compute_std(gyro_y_data, WINDOW);
    float t56 = micros() - start;

    // --- f57 ---
    start = micros();
    float f57 = compute_std(gyro_z_data, WINDOW);
    float t57 = micros() - start;

    // --- f58 ---
    start = micros();
    float f58 = compute_iqr(acc_x_data, WINDOW);
    float t58 = micros() - start;

    // --- f59 ---
    start = micros();
    float f59 = compute_iqr(acc_y_data, WINDOW);
    float t59 = micros() - start;

    // --- f60 ---
    start = micros();
    float f60 = compute_iqr(acc_z_data, WINDOW);
    float t60 = micros() - start;

    // --- f61 ---
    start = micros();
    float f61 = compute_iqr(gyro_x_data, WINDOW);
    float t61 = micros() - start;

    // --- f62 ---
    start = micros();
    float f62 = compute_iqr(gyro_y_data, WINDOW);
    float t62 = micros() - start;

    // --- f63 ---
    start = micros();
    float f63 = compute_iqr(gyro_z_data, WINDOW);
    float t63 = micros() - start;

    // --- f64 ---
    start = micros();
    compute_FFT_real(acc_x_data, fft_real_acc_x, WINDOW);
    float f64 = fft_real_acc_x[1]; // first non-DC bin
    float t64 = micros() - start;

    // --- f65 ---
    start = micros();
    compute_FFT_real(acc_y_data, fft_real_acc_y, WINDOW);
    float f65 = fft_real_acc_y[1];
    float t65 = micros() - start;

    // --- f66 ---
    start = micros();
    compute_FFT_real(acc_z_data, fft_real_acc_z, WINDOW);
    float f66 = fft_real_acc_z[1]; //  float f66 = fft_real_acc_z[8]; // first non-DC bin
    float t66 = micros() - start;

    // --- f67 ---
    start = micros();
    compute_FFT_real(gyro_x_data, fft_real_gyro_x, WINDOW);
    float f67 = fft_real_gyro_x[1];
    float t67 = micros() - start;

    // --- f68 ---
    start = micros();
    compute_FFT_real(gyro_y_data, fft_real_gyro_y, WINDOW);
    float f68 = fft_real_gyro_y[1];
    float t68 = micros() - start;

    // --- f69 ---
    start = micros();
    compute_FFT_real(gyro_z_data, fft_real_gyro_z, WINDOW);
    float f69 = fft_real_gyro_z[1];
    float t69 = micros() - start;

    // --- f70 ---
    start = micros();
    compute_FFT_mag(acc_x_data, fft_mag_acc_x, WINDOW);
    float f70 = compute_fft_energy(fft_mag_acc_x, WINDOW);
    float t70 = micros() - start;

    // --- f71 ---
    start = micros();
    compute_FFT_mag(acc_y_data, fft_mag_acc_y, WINDOW);
    float f71 = compute_fft_energy(fft_mag_acc_y, WINDOW);
    float t71 = micros() - start;

    // --- f72 ---
    start = micros();
    compute_FFT_mag(acc_z_data, fft_mag_acc_z, WINDOW);
    float f72 = compute_fft_energy(fft_mag_acc_z, WINDOW);
    float t72 = micros() - start;

    // --- f73 ---
    start = micros();
    compute_FFT_mag(gyro_x_data, fft_mag_gyro_x, WINDOW);
    float f73 = compute_fft_energy(fft_mag_gyro_x, WINDOW);
    float t73 = micros() - start;

    // --- f74 ---
    start = micros();
    compute_FFT_mag(gyro_y_data, fft_mag_gyro_y, WINDOW);
    float f74 = compute_fft_energy(fft_mag_gyro_y, WINDOW);
    float t74 = micros() - start;

    // --- f75 ---
    start = micros();
    compute_FFT_mag(gyro_z_data, fft_mag_gyro_z, WINDOW);
    float f75 = compute_fft_energy(fft_mag_gyro_z, WINDOW);
    float t75 = micros() - start;

    // Print features

    float values[] = {f1,  f2,  f3,  f4,  f5,  f6,  f7,  f8,  f9,  f10, f11, f12, f13, f14, f15, f16, f17, f18, f19,
                      f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38,
                      f39, f40, f41, f42, f43, f44, f45, f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57,
                      f58, f59, f60, f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75};

    printCSV(values, sizeof(values) / sizeof(values[0]));

    float times[] = {t1,  t2,  t3,  t4,  t5,  t6,  t7,  t8,  t9,  t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
                     t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38,
                     t39, t40, t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57,
                     t58, t59, t60, t61, t62, t63, t64, t65, t66, t67, t68, t69, t70, t71, t72, t73, t74, t75};
    Serial.println();
    Serial.println("Computation times (micro sec):");
    printCSV(times, sizeof(times) / sizeof(times[0]));
    Serial.println();
}

void
loop() {
    // run once
}