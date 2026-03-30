#include "lego_motor.h"

// The GPIO port JP1 on the DE1-SoC lives at this memory address.
// Writing to this address is how we talk to anything plugged into the 40-pin header.
#define GPIO_BASE 0xFF200060

// This is the actual data register — writing here changes what the pins output.
// "volatile" tells the compiler "don't optimize this away, the hardware is watching"
#define GPIO_DATA (*(volatile unsigned int *)(GPIO_BASE))

// This is the direction register — a 1 means output, a 0 means input.
// We need to tell the chip which pins WE control vs which ones we read from.
#define GPIO_DIR  (*(volatile unsigned int *)(GPIO_BASE + 0x4))

// ─────────────────────────────────────────────────────────────────────────────
// MOTOR ENABLE BITS
// According to the LEGO controller doc, each motor has two GPIO pins:
// one to enable/disable it, and one to set direction.
//
// IMPORTANT: 0 means ON and 1 means OFF for the enable pin. Yes, it's backwards.
// That's just how this hardware was designed — you "pull the pin low" to turn it on.
// ─────────────────────────────────────────────────────────────────────────────

// Motor 0's enable pin lives at GP0, which is bit 0 of the GPIO register
#define MOTOR0_EN  (1 << 0)

// Motor 1's enable pin is GP2 — bit 2
#define MOTOR1_EN  (1 << 2)

// Motor 2's enable pin is GP4 — bit 4
#define MOTOR2_EN  (1 << 4)

// Motor 3's enable pin is GP6 — bit 6
#define MOTOR3_EN  (1 << 6)

// Motor 4's enable pin is GP8 — bit 8
#define MOTOR4_EN  (1 << 8)

// ─────────────────────────────────────────────────────────────────────────────
// MOTOR DIRECTION BITS
// 0 = clockwise, 1 = counter-clockwise.
// Again, a bit counterintuitive, but that's what the doc says.
// ─────────────────────────────────────────────────────────────────────────────

// Motor 0's direction pin is GP1 — bit 1 (right next to its enable pin)
#define MOTOR0_DIR (1 << 1)

// Motor 1's direction pin is GP3 — bit 3
#define MOTOR1_DIR (1 << 3)

// Motor 2's direction pin is GP5 — bit 5
#define MOTOR2_DIR (1 << 5)

// Motor 3's direction pin is GP7 — bit 7
#define MOTOR3_DIR (1 << 7)

// Motor 4's direction pin is GP9 — bit 9
#define MOTOR4_DIR (1 << 9)

// This is a bitmask covering all 10 motor pins (GP0 through GP9).
// We use this to set all of them as outputs in one shot during setup.
// 0x3FF in binary is 0000 0011 1111 1111 — exactly bits 0 through 9.
#define ALL_MOTOR_PINS 0x3FF

// ─────────────────────────────────────────────────────────────────────────────
// DELAY FUNCTION
// The Nios V has no built-in sleep() so we just burn CPU cycles.
// How long this actually takes depends on your clock speed —
// you'll probably need to tune the number you pass in during testing.
// ─────────────────────────────────────────────────────────────────────────────
void delay(int how_long) {

    // We need "volatile" here so the compiler doesn't look at this empty loop
    // and think "that does nothing, I'll remove it". The volatile forces it to keep it.
    volatile int i;

    // Just count up to whatever number was passed in — higher number = longer wait
    for (i = 0; i < how_long; i++);
}

// ─────────────────────────────────────────────────────────────────────────────
// STOP ALL MOTORS
// Call this at startup, or any time something goes wrong and you need
// everything to stop immediately. Sets all enable pins HIGH (which means OFF).
// ─────────────────────────────────────────────────────────────────────────────
void stop_all_motors() {

    // OR-ing in all the enable bits forces them all to 1, which disables every motor.
    // We use |= so we don't accidentally wipe out any other bits in the register.
    GPIO_DATA |= (MOTOR0_EN | MOTOR1_EN | MOTOR2_EN | MOTOR3_EN | MOTOR4_EN);
}

// ─────────────────────────────────────────────────────────────────────────────
// GPIO SETUP
// Has to run once before you do anything else.
// Tells the hardware which pins are outputs and makes sure nothing is spinning.
// ─────────────────────────────────────────────────────────────────────────────
void setup_gpio() {

    GPIO_DATA = ALL_MOTOR_PINS;
    // Set GP0 through GP9 as outputs by writing 1s into the direction register.
    // We use |= so we don't accidentally mess with bits beyond GP9.
    GPIO_DIR |= ALL_MOTOR_PINS;
    GPIO_DATA = ALL_MOTOR_PINS;
    // Now disable all motors as a safety measure — we don't want anything
    // randomly spinning the moment the board powers up.
    stop_all_motors();
}

// ─────────────────────────────────────────────────────────────────────────────
// SPIN A MOTOR
// Pass in which motor (0 to 4) and which direction (1 = clockwise, 0 = counter).
// This is the function you'll call from your gate trigger when the CNN fires.
// ─────────────────────────────────────────────────────────────────────────────
void spin_motor(int motor_number, int go_clockwise) {

    // Figure out which bit controls this motor's enable pin.
    // Motor 0 → bit 0, Motor 1 → bit 2, Motor 2 → bit 4, etc.
    // Multiplying by 2 skips every other bit because each motor takes up 2 bits.
    unsigned int enable_bit = (1 << (motor_number * 2));

    // The direction bit is always the one right after the enable bit.
    // Motor 0 → bit 1, Motor 1 → bit 3, Motor 2 → bit 5, etc.
    unsigned int direction_bit = (1 << (motor_number * 2 + 1));

    // Read whatever is currently in the GPIO register.
    // We do this so we can change just this motor's bits without
    // accidentally turning off a motor that's already running.
    unsigned int current_state = GPIO_DATA;

    // Clear the enable bit for this motor (set it to 0).
    // Remember: 0 = enabled on this hardware. Clearing the bit turns the motor ON.
    // The ~ flips all bits of enable_bit, then & clears just that one bit.
    current_state &= ~enable_bit;

    // Now set the direction based on what was passed in
    if (go_clockwise) {

        // Clockwise = direction bit set to 0, so we clear it
        current_state &= ~direction_bit;

    } else {

        // Counter-clockwise = direction bit set to 1, so we set it
        current_state |= direction_bit;
    }

    // Write our updated value back to the hardware — this is what actually moves the motor
    GPIO_DATA = current_state;
}

// ─────────────────────────────────────────────────────────────────────────────
// STOP ONE MOTOR
// Stops a specific motor without touching anything else.
// ─────────────────────────────────────────────────────────────────────────────
void stop_motor(int motor_number) {

    // Calculate the enable bit for this motor (same formula as above)
    unsigned int enable_bit = (1 << (motor_number * 2));

    // Set that bit to 1, which disables (stops) this motor.
    // Again using |= so we leave all other motors exactly as they were.
    GPIO_DATA |= enable_bit;
}

/* main() removed — motor functions are called from main.c */
