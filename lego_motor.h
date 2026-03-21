#ifndef LEGO_MOTOR_H
#define LEGO_MOTOR_H

void setup_gpio(void);
void spin_motor(int motor_number, int go_clockwise);
void stop_motor(int motor_number);
void stop_all_motors(void);
void delay(int how_long);

#endif /* LEGO_MOTOR_H */
