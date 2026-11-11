# JetRacer Simulator: connect to the Spark and start it

Your group works on one of three DGX Spark computers. Your instructor tells you **which Spark**, your
**group number** and your **password**.

| Group | Account | Simulator port |
|---|---|---|
| 01 | `finalproject-group01` | `8701` |
| 02 | `finalproject-group02` | `8702` |
| 03 | `finalproject-group03` | `8703` |
| 04 | `finalproject-group04` | `8704` |
| 05 | `finalproject-group05` | `8705` |
| 06 | `finalproject-group06` | `8706` |

You need a terminal on your computer: **PowerShell** on Windows, **Terminal** on macOS or Linux.

## 1. Connect

Type this in the terminal on your computer, with your port, account and Spark address:

```
ssh -L PORT:localhost:PORT ACCOUNT@ADDRESS
```

For example, group 03 on Spark 2:

```
ssh -L 8703:localhost:8703 finalproject-group03@128.235.248.197
```

- The first time, it asks `Are you sure you want to continue connecting?`. Type `yes`.
- Type your password and press Enter. Nothing shows while you type: that is normal.
- You are in when the prompt shows `finalproject-group03@spark-...:~$`.

The `-L 8703:localhost:8703` part is a tunnel: it lets the browser on your computer reach the simulator on
the Spark.

## 2. Start the simulator

In the same terminal:

```
jetracer-sim
```

Wait for `JetRacer Simulator running at http://localhost:8703`. Keep this terminal open while you work.

It also prints a tunnel command (`ssh -N -L ...`). You don't need it: your login from step 1 already has the
tunnel.

## 3. Open it in your browser

Go to **http://localhost:8703** (your group's port) on your computer.

Then follow the tabs: **Drive**, **Collect**, **Train**, **Self-Drive**. Press `?` in the app for the
keyboard shortcuts.

Teammates can watch the same simulator: each logs in with the same command from step 1 and opens the same
address. Only one person runs `jetracer-sim`.

## 4. Stop

- **Ctrl+C** in the terminal stops the simulator.
- `exit` logs you out.

Your images and models stay saved in your account, in `~/jetracer/notebooks`.

All groups share one GPU. Close the browser tab when you are not using the simulator.

## If something goes wrong

| You see | What to do |
|---|---|
| `Permission denied` | Wrong password, account or Spark address. |
| `Connection timed out` | Check the address. Off campus, you may need the university VPN. |
| The browser can't reach the page, and the terminal shows `connect failed: Connection refused` | The simulator isn't running yet. Run `jetracer-sim` and wait for "running". |
| `The JetRacer Simulator is already running at http://localhost:8703` | Your group's simulator is already running (a teammate, or an old session). Just open the page. To restart it: `pkill -u $USER -f start_simulator.py`, then `jetracer-sim`. |
| `port 8703 is already used by another program ...` | Tell your instructor. |
| `Address already in use` when you log in | Another terminal on your computer is already connected with the tunnel. Close it, then log in again. |
