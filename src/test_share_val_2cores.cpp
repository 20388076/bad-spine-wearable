#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <Arduino.h>           // Core Arduino functions

volatile int predicted = -1;        // Shared variable
SemaphoreHandle_t xMutex;           // Mutex handle

void Task1code(void *pvParameters) {
    for (;;) {
        int localPred = random(0, 5); // Simulate model.predict(...)
        xSemaphoreTake(xMutex, portMAX_DELAY);
        predicted = localPred;
        xSemaphoreGive(xMutex);
        vTaskDelay(pdMS_TO_TICKS(1000)); // 1 Hz update rate
    }
}

void Task2code(void *pvParameters) {
    for (;;) {
        int localCopy;
        xSemaphoreTake(xMutex, portMAX_DELAY);
        localCopy = predicted;
        xSemaphoreGive(xMutex);

        Serial.printf("Core 1 sees prediction: %d\n", localCopy);
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

void setup() {
    Serial.begin(115200);
    xMutex = xSemaphoreCreateMutex();

    xTaskCreatePinnedToCore(Task1code, "Task1", 4096, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(Task2code, "Task2", 4096, NULL, 1, NULL, 1);
}

void loop() {}