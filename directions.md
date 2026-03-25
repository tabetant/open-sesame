# Running open-sesame on the DE1-SoC

All commands are run from **Windows PowerShell** on the DESL lab computers.
You need your project folder open — navigate there first:
```powershell
cd C:\path\to\open-sesame
```

---

## Step 1 — Flash the FPGA

Do this once each time the board is powered on.
```powershell
./gmake DE1-SoC
```
This loads the Nios V computer system bitstream onto the Cyclone V FPGA.

If it fails, run:
```powershell
./gmake DETECT_DEVICES
```
If the two JTAG devices are listed in reverse order (SOCVHPS before Cyclone V), open the Makefile and change:
```
JTAG_INDEX_SoC := 2
```
to `1`, then retry.

---

## Step 2 — Compile

```powershell
./gmake COMPILE
```
Produces `main.elf`. Fix any errors before continuing.

---

## Step 3 — Open 3 PowerShell windows

You need them all open and navigated to the project folder.

### Window A — GDB Server (keep this running)
```powershell
./gmake GDB_SERVER
```
Leave it. It will block and print something like:
```
Waiting for connection on port 2454...
```

### Window B — JTAG Terminal (open before running the program)
```powershell
./gmake TERMINAL
```
This runs `nios2-terminal.exe --instance 0`. It will sit silently until the program starts sending output. Leave it open — this is where you'll see:
```
=== Open Sesame Gate ===
Audio codec initialised. LISTENING...
```

### Window C — Load and run
```powershell
./gmake GDB_CLIENT
```
GDB connects, loads `main.elf` onto the board, and sets the program counter to `_start`.
You'll land at a `(gdb)` prompt. Type:
```
continue
```
The program starts running. Switch to Window B to watch the output.

To run it in the background (so you can still type GDB commands):
```
continue &
```

---

## What you should see

**LEDs on the board:**
- `LED[0]` lights up at boot
- `LED[1]` pulses every ~0.1 s while audio is flowing
- After each inference: LEDs 5–9 show the confidence level
- If all 10 LEDs go dark (except `LED[0]`), that's the idle reset — normal

**Window B (JTAG terminal), once per second:**
```
PROCESSING...
  MFCC extraction done
  CNN inference done
  Confidence: 0.12
LISTENING...
```
If it hears "open sesame" with >90% confidence:
```
  *** TRIGGERED! ***
```

**VGA display (if a monitor is connected):**
- Row 0: `=== Open Sesame Gate ===`
- Row 1: live status (same stages as the JTAG output)

---

## Stopping / restarting

In the GDB client (Window C):
```
interrupt       # pause the running program
continue        # resume
set $pc=_start  # restart from the beginning
quit            # exit GDB (also stops the program)
```
After quitting GDB, also stop the GDB server in Window A with `Ctrl+C`.
Close the terminal in Window B with `Ctrl+C`.

---

## Recompile and reload (normal dev loop)

```powershell
# Window C — interrupt and quit GDB first
interrupt
quit

# Window A — Ctrl+C to stop GDB server, then restart it
./gmake GDB_SERVER

# Any window — recompile
./gmake COMPILE

# Window C — reload
./gmake GDB_CLIENT
# then: continue
```
You do NOT need to re-flash the FPGA (`./gmake DE1-SoC`) unless the board was power-cycled.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `./gmake DE1-SoC` fails | Run `./gmake DETECT_DEVICES`, check JTAG device order, adjust `JTAG_INDEX_SoC` in Makefile |
| Window B shows nothing | Make sure you typed `continue` in the GDB client, and that the GDB server is still running |
| `LED[1]` not pulsing | Audio codec init may have failed — power-cycle the board and re-flash |
| Program hangs (LEDs frozen) | The last lit LED shows where it stopped (see LED map in CLAUDE.md) |
| GDB client can't connect | GDB server must be started first in Window A |
