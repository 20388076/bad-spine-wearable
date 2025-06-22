#ifndef FFT_REAL_H
#define FFT_REAL_H

#include <Arduino.h>
#include <math.h>

// Custom FFT real part implementation (mimicking numpy's np.real(fft(x)))
class FFT_real {
  public:
    FFT_real(int n) : N(n) {
        input = new float[N];
        output = new float[N];
    }

    ~FFT_real() {
        delete[] input;
        delete[] output;
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
            float real = 0.0;
            for (int n = 0; n < N; ++n) {
                float angle = -2.0 * PI * k * n / N;
                real += input[n] * cos(angle);
            }
            output[k] = real;
        }
    }

    float*
    getOutput() {
        return output;
    }

  private:
    int N;
    float* input;
    float* output;
};

#endif