#include <ESP32Servo.h>

Servo myServo;         // Create Servo object
int servoPin = 26;     // Choose a GPIO pin for PWM control
int currentAngle = 40; // Start at the center position

void
setup() {
    myServo.attach(servoPin);    // Attach the servo
    myServo.write(currentAngle); // Move to initial position
    randomSeed(analogRead(0));   // Seed random values
}

void
loop() {
    myServo.write(currentAngle); // Move to initial position
}