#include <Arduino.h>
#include <Wire.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
TaskHandle_t Task1, Task2;
void Task1code(void* pvParameters);
void Task2code(void* pvParameters);
TaskFunction_t Task1code1, Task1code2;

void
loop() {}

void
setup() {
    Serial.begin(115200);

    //create a task that will be executed in the Task1code() function, with priority 1 and executed on core 0
    xTaskCreatePinnedToCore(Task1code, /* Task function. */
                            "Task1",   /* name of task. */
                            10000,     /* Stack size of task */
                            NULL,      /* parameter of the task */
                            1,         /* priority of the task */
                            &Task1,    /* Task handle to keep track of created task */
                            0);        /* pin task to core 0 */
    delay(500);

    //create a task that will be executed in the Task2code() function, with priority 1 and executed on core 1
    xTaskCreatePinnedToCore(Task2code, /* Task function. */
                            "Task2",   /* name of task. */
                            10000,     /* Stack size of task */
                            NULL,      /* parameter of the task */
                            1,         /* priority of the task */
                            &Task2,    /* Task handle to keep track of created task */
                            1);        /* pin task to core 1 */
    delay(500);
}

//Task1code: core0 test
void
Task1code(void* pvParameters) {
    for (;;) {
        Serial.print("Task1 running on core ");
        Serial.println(xPortGetCoreID());
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

//Task2code: core1 test
void
Task2code(void* pvParameters) {
    for (;;) {
        Serial.print("Task2 running on core ");
        Serial.println(xPortGetCoreID());
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
