/*
 * lego_motor.c — LEGO Motor Control via GPIO
 *
 * ── Overview ──────────────────────────────────────────────────────────────────
 *
 * This file controls up to 5 LEGO motors by toggling GPIO (General Purpose
 * Input/Output) pins on the DE1-SoC's JP1 expansion header.
 *
 * GPIO means "pins you can control in software".  You tell the hardware whether
 * each pin is an input (reading a sensor) or an output (driving a device), then
 * write 0s and 1s to the output pins.  Those voltage levels (0V or 3.3V) drive
 * a motor controller chip that powers the LEGO motors.
 *
 * ── Hardware wiring ───────────────────────────────────────────────────────────
 *
 * The JP1 GPIO header at address 0xFF200060 has 36 pins.  We use the first 10:
 *
 *   Pin (GP0, GP1):  Motor 0 enable + direction
 *   Pin (GP2, GP3):  Motor 1 enable + direction
 *   Pin (GP4, GP5):  Motor 2 enable + direction
 *   Pin (GP6, GP7):  Motor 3 enable + direction
 *   Pin (GP8, GP9):  Motor 4 enable + direction
 *
 * For each motor, there are two signals:
 *   ENABLE  (even bit): 0 = motor ON,  1 = motor OFF  ← active-low
 *   DIR     (odd  bit): 0 = clockwise, 1 = counter-clockwise
 *
 * Note the inverted logic on the ENABLE pin!  This is called "active-low".
 * The motor controller chip has an inverted enable input, which is common in
 * hardware design for historical and noise-immunity reasons.
 *
 * ── Register map ──────────────────────────────────────────────────────────────
 *
 * GPIO_DATA (offset 0x00):
 *   Reading  → current pin state
 *   Writing  → drive the output pins to these values
 *
 * GPIO_DIR  (offset 0x04):
 *   Bit = 1 → that pin is an OUTPUT (we drive it)
 *   Bit = 0 → that pin is an INPUT  (we read it)
 *   We set bits 0–9 to 1 because all motor pins are outputs.
 *
 * ── The motor is NOT YET WIRED on this branch ─────────────────────────────────
 * open_and_close_gate() in main.c is defined but the call is commented out.
 * These functions are ready for when the motor is connected.
 */

#include "lego_motor.h"

/*
 * JP1 GPIO base address.
 * This is the memory address where the GPIO peripheral's registers live.
 * Reading or writing to this address talks directly to the hardware pins.
 */
#define GPIO_BASE 0xFF200060

/*
 * GPIO_DATA — the data register.
 *
 * Writing a 1 to bit N drives pin GP(N) high (3.3 V).
 * Writing a 0 to bit N drives pin GP(N) low  (0 V).
 * Reading gives the current state of all pins.
 *
 * "volatile" tells the compiler: "do not optimise away reads/writes to this
 * address — the hardware can change it at any time independently of the CPU."
 * Without volatile, the compiler might decide "I already know the value, I'll
 * skip the read" — which would be wrong for hardware registers.
 */
#define GPIO_DATA (*(volatile unsigned int *)(GPIO_BASE))

/*
 * GPIO_DIR — the direction register.
 *
 * Set a bit to 1 to make that pin an output (we control it).
 * Set a bit to 0 to make that pin an input  (we read it).
 * We set bits 0–9 as outputs so we can drive the motor controller.
 */
#define GPIO_DIR  (*(volatile unsigned int *)(GPIO_BASE + 0x4))

/* ── Motor pin bit masks ──────────────────────────────────────────────────────
 *
 * Each motor uses two consecutive bits:
 *   Bit (motor_number * 2)     = ENABLE  (0 = ON, 1 = OFF — active-low)
 *   Bit (motor_number * 2 + 1) = DIR     (0 = CW, 1 = CCW)
 *
 * These constants are the bitmasks for each individual pin.
 * (1 << N) = a 32-bit integer with only bit N set.
 *
 * Motor 0: GP0 (bit 0) = enable, GP1 (bit 1) = direction
 * Motor 1: GP2 (bit 2) = enable, GP3 (bit 3) = direction
 * etc.
 */

/* Enable pin masks — set to 1 to DISABLE the motor (active-low) */
#define MOTOR0_EN  (1 << 0)   /* GP0: motor 0 enable */
#define MOTOR1_EN  (1 << 2)   /* GP2: motor 1 enable */
#define MOTOR2_EN  (1 << 4)   /* GP4: motor 2 enable */
#define MOTOR3_EN  (1 << 6)   /* GP6: motor 3 enable */
#define MOTOR4_EN  (1 << 8)   /* GP8: motor 4 enable */

/* Direction pin masks — set to 1 for counter-clockwise */
#define MOTOR0_DIR (1 << 1)   /* GP1: motor 0 direction */
#define MOTOR1_DIR (1 << 3)   /* GP3: motor 1 direction */
#define MOTOR2_DIR (1 << 5)   /* GP5: motor 2 direction */
#define MOTOR3_DIR (1 << 7)   /* GP7: motor 3 direction */
#define MOTOR4_DIR (1 << 9)   /* GP9: motor 4 direction */

/*
 * ALL_MOTOR_PINS — bitmask for all 10 motor control pins (bits 0–9).
 *
 * 0x3FF = 0b 0011 1111 1111 = bits 0 through 9 all set.
 * Used to configure all motor pins as outputs in one write to GPIO_DIR,
 * and to disable all motors at startup.
 */
#define ALL_MOTOR_PINS 0x3FF


/*
 * delay — busy-wait for approximately 'how_long' CPU cycles.
 *
 * The Nios V has no hardware sleep timer, so we waste CPU cycles in a loop.
 *
 * The "volatile" on the loop counter prevents the compiler from removing the
 * loop entirely.  Without volatile, the compiler sees an empty loop and
 * optimises it away to nothing (since the loop body has no side effects).
 *
 * Timing:
 *   At ~100 MHz, one loop iteration ≈ 10 ns.
 *   delay(50000000) ≈ 500 ms (0.5 seconds) — used for OPEN_DELAY.
 *   delay(20000000) ≈ 200 ms (0.2 seconds) — used for HOLD_DELAY.
 *
 * These estimates are rough.  The actual time depends on the exact Nios V
 * clock frequency and whether the loop is in cache.  Tune the constants in
 * main.c by testing on real hardware with a stopwatch.
 */
void delay(int how_long)
{
    volatile int i;
    for (i = 0; i < how_long; i++);  /* empty loop body — just burns time */
}


/*
 * stop_all_motors — immediately disable all 5 motors.
 *
 * Sets all ENABLE pins HIGH (logic 1 = motor off, because active-low).
 * Uses OR-assignment so we only change the enable bits, leaving the direction
 * bits untouched.  Safe to call at any time as an emergency stop.
 *
 * Called at startup before anything else to ensure no motor is accidentally
 * spinning when the board powers on or the program is loaded.
 */
void stop_all_motors(void)
{
    /* OR-in all enable bits to set them to 1 (disabled).
     * This does NOT change any direction bits or non-motor GPIO bits. */
    GPIO_DATA |= (MOTOR0_EN | MOTOR1_EN | MOTOR2_EN | MOTOR3_EN | MOTOR4_EN);
}


/*
 * setup_gpio — one-time GPIO initialisation.
 *
 * Must be called once at the start of main() before any motor commands.
 * Two tasks:
 *   1. Set bits 0–9 as OUTPUT pins so we can drive the motor controller.
 *   2. Call stop_all_motors() to ensure nothing spins at power-on.
 *
 * Uses OR-assignment on GPIO_DIR so we only change bits 0–9, preserving
 * any other GPIO configuration that might have been set elsewhere.
 */
void setup_gpio(void)
{
    /* Set GP0 through GP9 as outputs.
     * |= means "set these bits to 1 without changing any other bits". */
    GPIO_DIR |= ALL_MOTOR_PINS;

    /* Safety: disable all motors before anything starts running */
    stop_all_motors();
}


/*
 * spin_motor — start one motor in a specified direction.
 *
 * Parameters:
 *   motor_number — which motor to spin (0 to 4)
 *   go_clockwise — 1 = clockwise, 0 = counter-clockwise
 *
 * How it works:
 *   1. Compute which GPIO bits control this motor.
 *   2. Read the current GPIO_DATA to avoid disturbing other motors.
 *   3. Clear the ENABLE bit (set to 0 = ON, because active-low).
 *   4. Set the DIRECTION bit based on go_clockwise.
 *   5. Write the modified value back to GPIO_DATA.
 *
 * Bit positions for motor N:
 *   enable    = bit (N * 2)
 *   direction = bit (N * 2 + 1)
 *
 * Example for motor 0 (N=0):
 *   enable    = bit 0, direction = bit 1
 * Example for motor 2 (N=2):
 *   enable    = bit 4, direction = bit 5
 */
void spin_motor(int motor_number, int go_clockwise)
{
    /* Compute the bitmask for this motor's enable pin.
     * motor 0 → bit 0, motor 1 → bit 2, motor 2 → bit 4, etc. */
    unsigned int enable_bit    = (1 << (motor_number * 2));

    /* Direction bit is always one bit above the enable bit */
    unsigned int direction_bit = (1 << (motor_number * 2 + 1));

    /* Read current state of all GPIO pins.
     * We do a read-modify-write to avoid accidentally stopping other motors. */
    unsigned int current_state = GPIO_DATA;

    /* Clear the enable bit → 0 → motor ON (active-low logic).
     * ~enable_bit flips all bits of the mask (so the enable bit becomes 0,
     * all others become 1).  AND-ing with this clears just that one bit. */
    current_state &= ~enable_bit;

    /* Set direction based on the parameter */
    if (go_clockwise) {
        /* Clockwise = direction bit = 0.  Clear it with the same AND trick. */
        current_state &= ~direction_bit;
    } else {
        /* Counter-clockwise = direction bit = 1.  Set it with OR. */
        current_state |= direction_bit;
    }

    /* Write the modified register value back — this actually moves the motor */
    GPIO_DATA = current_state;
}


/*
 * stop_motor — stop one specific motor without affecting any others.
 *
 * Sets only the ENABLE bit for the specified motor to 1 (= disabled).
 * Uses OR-assignment so other motors keep running if they were already on.
 *
 * Parameter:
 *   motor_number — which motor to stop (0 to 4)
 */
void stop_motor(int motor_number)
{
    /* Enable bit for this motor (same formula as spin_motor) */
    unsigned int enable_bit = (1 << (motor_number * 2));

    /* Set that bit to 1 (disabled).  OR-assign leaves all other bits alone. */
    GPIO_DATA |= enable_bit;
}
