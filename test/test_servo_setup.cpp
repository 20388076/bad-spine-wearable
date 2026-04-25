#include <Arduino.h>    // Core Arduino functions
#include <ESP32Servo.h> // Library to control servo motors on ESP32
Servo myServo;          // Create Servo object
int servoPin = 25;      // Choose a GPIO pin for PWM control. Connect servo to pin 25 or D2
int currentAngle = 90;  // Start at the center position. Connect servo to pin 25 or D2

void
setup() {
    myServo.attach(servoPin);    // Attach the servo
    myServo.write(currentAngle); // Move to initial position
}

void
loop() {
    myServo.write(currentAngle); // Move to initial position
}