#ifndef FFT_REAL_H
#define FFT_REAL_H

#include <Arduino.h>
#include <math.h>

class FFT_real {
  public:
    FFT_real(int n) : N(n) {
        input = new float[N];
        real = new float[N];
        imag = new float[N];
        magnitude = new float[N];
    }

    ~FFT_real() {
        delete[] input;
        delete[] real;
        delete[] imag;
        delete[] magnitude;
    }

    void
    setInput(float* data) {
        for (int i = 0; i < N; ++i) {
            input[i] = data[i];
        }
    }

    void
    compute() {
        for (int k = 0; k < N; ++k) {
            float sum_real = 0.0;
            float sum_imag = 0.0;
            for (int n = 0; n < N; ++n) {
                float angle = 2.0 * PI * k * n / N;
                sum_real += input[n] * cos(angle);
                sum_imag -= input[n] * sin(angle); // Note minus for FFT
            }
            real[k] = sum_real;
            imag[k] = sum_imag;
        }
    }

    void
    computeMagnitude() {
        for (int i = 0; i < N; ++i) {
            magnitude[i] = sqrt(real[i] * real[i] + imag[i] * imag[i]);
        }
    }

    float*
    getReal() {
        return real;
    }

    float*
    getImag() {
        return imag;
    }

    float*
    getMagnitude() {
        return magnitude;
    }

  private:
    int N;
    float* input;
    float* real;
    float* imag;
    float* magnitude;
};

#endif
