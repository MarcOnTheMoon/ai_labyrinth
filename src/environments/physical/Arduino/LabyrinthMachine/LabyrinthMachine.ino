/*************************************************** 
  reads the serially transmitted message, divides the read string into individual components 
  and controls the servos according to the specifications.
  The Adafruit drivers use I2C to communicate (2 Pins: SDA, SCL).

@authors: Sandra Lassahn
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.08.20
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
 ****************************************************/
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h> 

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(); //default address 0x40
#define SERVO_FREQ 50 // Analog servos run at ~50 Hz updates

String command_message, channel_str, degree_str;
char msg[10];
int channel[] = {0, 1}; //Connected channel of the servo to the PWM driver, [y,x]
int pulse_width_max[] = {2540, 2650}; //The maximum and minimum pulse width is different for each servo
int pulse_width_min[] = {420, 350};
float pulse_middle[] = {((pulse_width_max[0] + pulse_width_min[0]) / 2), ((pulse_width_max[1] + pulse_width_min[1]) / 2)};
float pulse_per_degree[] = {((pulse_width_max[0] - pulse_middle[0])/ (asin(5.2 / 122.5) * 180 / M_PI)), ((pulse_width_max[1] - pulse_middle[1]) / (asin(8.6/155.5)*180 / M_PI))}; //Depending on the used servo
float degree[] = {0.0, 0.0};
float min_degree[] = {((pulse_width_min[0] - pulse_middle[0]) / pulse_per_degree[0]), ((pulse_width_min[1] - pulse_middle[1]) / pulse_per_degree[1])};
float max_degree[] = {((pulse_width_max[0] - pulse_middle[0]) / pulse_per_degree[0]), ((pulse_width_max[1] - pulse_middle[1]) / pulse_per_degree[1])};

void setup() {
  Serial.begin(115200);  // Start serial communication with baudrate 115200
  Serial.setTimeout(2);  // Timeout in ms - Waiting time until no more characters are read in

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);  // Analog servos run at ~50 Hz updates

  delay(10);
}

void loop() {
  while (!Serial.available());

  //Message lesen 
  command_message = Serial.readString();
  command_message.toCharArray(msg, command_message.length()+1);   // conversion to charArray is needed for strtok

  channel_str = strtok(msg,";"); //Numbers before semicolons
  int channel = atoi(channel_str.c_str()); // convert char array to int
  degree_str = strtok(NULL, ";"); // take out the next number block after the semicolon
  float degree = atof(degree_str.c_str()); // convert char array to float

  int pulse_width = int(degree * pulse_per_degree[channel] + pulse_middle[channel]); //Calculates the corresponding pulse length for the desired playing field angle

  // Limiting pulse durations to the permissible range.
  if (pulse_width < pulse_width_min[channel]) {
    pulse_width = pulse_width_min[channel];
  } else if (pulse_width > pulse_width_max[channel]) {
    pulse_width = pulse_width_max[channel];
  }

  // Set PWM signal for the servo control
  Serial.print(pulse_width);
  pwm.writeMicroseconds(channel, pulse_width);
  delay(20); //Delay time in ms as security for servos
}
