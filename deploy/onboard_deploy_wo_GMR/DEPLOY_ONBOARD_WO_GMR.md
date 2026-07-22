# Onboard Deployment (Cable-Free, Workstation-Assisted GMR)

This guide describes the third deployment mode of **Humanoid-GPT**, sitting
between the two existing options:

| Mode                                                     | Mocap + GMR runs on  | Tracking ONNX runs on | Cable required?              |
|----------------------------------------------------------|----------------------|-----------------------|------------------------------|
| `deploy/play_track.py`                                   | Workstation          | Workstation           | **Yes** (Ethernet to G1)     |
| `deploy/onboard_deploy/play_track_onboard.py`            | G1 Jetson            | G1 Jetson             | No, but Jetson CPU saturates |
| `deploy/onboard_deploy_wo_GMR/play_track_onboard_wo_GMR` | **4090 workstation** | **G1 Jetson**         | **No**                       |

The new mode keeps the cable-free convenience of the onboard build while
offloading the CPU-hungry Noitom client and GMR (IK) retargeting to a
dedicated 4090 workstation.  The G1 then only needs to run the lightweight
MuJoCo FK + ONNX tracking + DDS motor control loop.

## Architecture

```
+-------------------------------- 4090 Workstation --------------------------------+
| Noitom PNLink (WiFi)  OR  Xsens MVN Network Streamer (TCP/UDP, port 9763)        |
|         |                              |                                         |
|         v                              v                                         |
| NoitomClient                      XsensClient                                    |
|         |  (only one is active at a time -- selected by --mocap-type)            |
|         v                                                                        |
| GeneralMotionRetargeting (IK) -> EMA -> UDP send (~90 Hz, ~180B)                 |
+----------------------------------------|-----------------------------------------+
                                         |  shared WiFi (same SSID/subnet)
                                         v
+--------------------------------- Unitree G1 Jetson ------------------------------+
| netmocap-rx subprocess  (core 2, SCHED_FIFO 40)                                  |
|   recvfrom + decode_frame -> shared-memory latch (mp.Array, single lock)         |
|         |                                                                        |
|         v                                                                        |
| Main process (loco thread, core 4, SCHED_FIFO 50)                                |
|   shared-mem read -> LiveRefConverter (MuJoCo FK)                                |
|         |                                                                        |
|         v                                                                        |
| ONNX / TRT tracking policy                                                       |
|         |                                                                        |
|         v                                                                        |
| DDS motor commands (eth0 internal bus)                                           |
+----------------------------------------------------------------------------------+
```

**Why this is fast on the Jetson:**

- No `NoitomClient` (no PNLink decode work).
- No `GeneralMotionRetargeting` (no per-frame IK).
- UDP recv runs in a dedicated subprocess pinned to core 2 with
  SCHED_FIFO priority 40, so the loco thread on core 4 is **never**
  blocked on the receiver's GIL (this was the source of audible motor
  "click" jitter in the earlier same-process-thread build).
- The shared-memory hand-off is the same `mp.Array + Lock` pattern used
  by the GMR-on-Jetson onboard deploy; reads are a single memcpy of the
  ~144 B qpos vector under a held-for-microseconds lock.

**Why this is robust over WiFi:**

- Packets are small (~180 B / frame, ~16 KB/s at 90 Hz).  A single packet
  is far below the 1500 B MTU so there is no IP fragmentation.
- UDP, not TCP -- a dropped packet is just dropped, never blocks the
  control loop and never causes retransmit storms.
- Each packet carries a sender wall-clock timestamp; the receiver drops
  out-of-order frames so the latch never moves backwards.

## Wire format (`protocol.py`)

Little-endian, fixed-layout header followed by float32 payload:

```
offset             size  field
0                  4     magic       b"HMCP"
4                  1     version     uint8 (=1)
5                  1     flags       uint8  bit0: has_hand    bit1: has_brainco
6                  4     seq         uint32  per-stream monotonic sequence
10                 8     send_ts     float64 sender time.time()
18                 2     n_qpos      uint16  (=36 for G1)
20                 4*n   qpos        float32 [root_pos3 | root_rot4 | dof29]
20+4*n             16    hand        float32 [l_open, l_dist, r_open, r_dist]
                                              (if has_hand)
20+4*n+[16]        96    brainco     float32 [left12 | right12]   (if has_brainco)
```

Packet sizes for the default G1 stream:

| Mode                                      | Size  |
|-------------------------------------------|-------|
| Dex3 (`has_hand=1, has_brainco=0`)        | 180 B |
| BrainCo (`has_hand=1, has_brainco=1`)     | 276 B |
| Body-only (`has_hand=0, has_brainco=0`)   | 164 B |

The BrainCo payload is the **24-D retargeted hand qpos** from GMR's
`brainco`/`brainco2`/`brainco3` target (left hand first, then right hand --
the exact layout `brainco_qpos24_to_cmd12` expects).  The receiver does the
12-D actuator conversion + EMA smoothing locally, so the wire format stays
stable across smoother / scale tweaks.

## Setup

### Workstation side (4090)

Any machine that already runs `deploy/play_track.py` works -- the
dependencies (`noitom`, `gmr`, `numpy`) are identical
because we re-use `deploy/retarget.py` as a subprocess.

```bash
conda activate h-gpt   # or whichever env has Noitom + GMR
cd ~/Humanoid-GPT
```

### Robot side (G1 Jetson)

Follow the existing on-board guide (`deploy/onboard_deploy/DEPLOY_ONBOARD.md`)
through **Step 5.3 Install Unitree SDK**.  In this new mode you do **not**
need to install Noitom or GMR on the robot at all:

```bash
# Skip these two -- only needed for the original on-board GMR path:
# pip install -e thirdparty/GMR-galbot
# pip install -e thirdparty/noitom
```

That alone saves a few hundred MB and trims dependencies on the Jetson.

### Network checklist

1. The workstation and the G1 are on the same WiFi (same SSID + subnet, e.g.
   both `192.168.1.0/24`).
2. From the workstation: `ping <g1_wifi_ip>` works.
3. From the G1: `ping <workstation_wifi_ip>` works.
4. UDP port `51234` is not blocked by a firewall on the G1.

## Running

### G1 Jetson runtime requirement

Before every real-robot run on the G1 Jetson, lock the Jetson clocks:

```bash
sudo jetson_clocks
```

Alternatively, flash the robot computer to **JetPack 6.2** and deploy with the TensorRT (`--use-trt`) runtime.

### 1) On the workstation

```bash
# Noitom Axis Studio (default):
python -m deploy.onboard_deploy_wo_GMR.host_sender \
    --robot-ip 192.168.1.42

# Xsens MVN (Network Streamer, TCP, default port 9763):
python -m deploy.onboard_deploy_wo_GMR.host_sender \
    --robot-ip 192.168.1.42 \
    --mocap-type xsens \
    --xsens-protocol tcp --xsens-port 9763 \
    --human-height 1.75
```

Useful flags:

| Flag                    | Default         | Description                                                                                                                                            |
|-------------------------|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--robot-ip`            | `192.168.1.42`  | G1 Jetson WiFi address.                                                                                                                                |
| `--robot-port`          | `51234`         | UDP port; must match the robot's `--listen-port`.                                                                                                      |
| `--mocap-type`          | `pnlink`        | `pnlink` or `xsens`.                                                                                                                                   |
| `--xsens-host`          | `0.0.0.0`       | Local bind address for the MVN MXTP02 listener.  Xsens-only.                                                                                           |
| `--xsens-port`          | `9763`          | Local port MVN Studio connects to.  Xsens-only; must match MVN Studio.                                                                                 |
| `--xsens-protocol`      | `tcp`           | `tcp` or `udp` — must match MVN Studio's Network Streamer.  Xsens-only.                                                                                |
| `--human-height`        | `1.7`           | Passed through to GMR for retargeting calibration.                                                                                                     |
| `--buffer-ms`           | `0.0`           | Host-side GMR jitter buffer. Keep 0 so each retargeted frame is forwarded ASAP.                                                                        |
| `--send-hz`             | `0.0`           | `0` = send-on-update (track GMR rate, ~90 Hz). Set e.g. `60` to cap on flaky WiFi.                                                                     |
| `--no-hand`             | off             | Strip the 4-float Dex3-style hand payload (open/dist).  Has no effect on BrainCo.                                                                      |
| `--enable-brainco-hand` | off             | Switch to the BrainCo-aware GMR subprocess and append a 24-D BrainCo hand qpos per packet.                                                             |
| `--hand-target`         | `brainco2`      | GMR hand target -- one of `brainco`, `brainco2`, `brainco3`.                                                                                           |
| `--visualize-retarget`  | off             | Also spawn a mujoco viewer on the workstation (works for Noitom/Xsens, and BrainCo).  Needs a display server -- don't enable over plain SSH.           |
| `--log-every-sec`       | `2.0`           | Stats print interval (`send Hz / lag / loss / bandwidth`).                                                                                             |

You'll see lines like:

```
[host_sender] send= 89.4Hz  sent=1789  dup_polls=2210  err=0  lag_ms mean= 2.31 max= 5.12  bw=  131.2 kbps
```

`lag_ms` is the wall-clock delay from GMR-output to `sendto` (does not
include the WiFi flight time).  Anything below ~5 ms is healthy.

### 2a) On the G1 (Dex3 hands or no hands)

```bash
sudo jetson_clocks

python -m deploy.onboard_deploy_wo_GMR.play_track_onboard_wo_GMR \
    --onnx-track storage/ckpts/pns_wo_priv216.onnx
```

Useful flags (the policy / DDS / hand flags are unchanged from the original
onboard deploy):

| Flag                      | Default          | Description                                                                          |
|---------------------------|------------------|--------------------------------------------------------------------------------------|
| `--listen-ip`             | `0.0.0.0`        | UDP bind address on the G1.                                                          |
| `--listen-port`           | `51234`          | UDP port; must match host_sender's `--robot-port`.                                   |
| `--startup-timeout-sec`   | `10.0`           | Wait this long for the first packet before starting up.                              |
| `--no-mocap`              | off              | Skip the UDP listener and only run walk + offline modes.                             |
| `--net-recv-rt-pin`       | `(2, 40)`        | `(cpu_id, SCHED_FIFO priority)` for the recv subprocess.  Pass `None` to disable.    |

### 2b) On the G1 (BrainCo dex hands)

Mirror image of (2a) but driving the BrainCo hands instead of Dex3.  Use
this when `host_sender` was launched with `--enable-brainco-hand`:

```bash
# Workstation
python -m deploy.onboard_deploy_wo_GMR.host_sender \
    --robot-ip 192.168.1.42 --enable-brainco-hand

# G1
sudo jetson_clocks

python -m deploy.onboard_deploy_wo_GMR.play_track_onboard_wo_GMR_brainco \
    --onnx-track storage/ckpts/pns_wo_priv216.onnx
```

Extra flags specific to the BrainCo build:

| Flag                          | Default | Description                                                                                                  |
|-------------------------------|---------|--------------------------------------------------------------------------------------------------------------|
| `--enable-brainco-hand`       | `True`  | Spin up `BraincoController` and drive it from the wire.  Set to `False` to leave the hands at rest.          |
| `--brainco-hand-fps`          | `100`   | Internal pub/sub rate inside `BraincoController`.                                                            |
| `--brainco-hand-smooth-alpha` | `0.45`  | EMA factor for the 12-D actuator command (smaller = smoother, larger = more responsive).                     |
| `--brainco-hand-scale`        | `1.0`   | Multiplier on the final 12-D command before clipping to [0,1].                                               |
| `--rest-hand-in-walk`         | `True`  | While the body is in walk mode (mode 0), drive the hands back to rest pose so they don't dangle.             |

This binary intentionally does **not** import `deploy.play_track` or
`deploy.brainco.play_track_brainco`, so the G1 does not need `pygame`,
`jax`, or `loop_rate_limiters`.  The small BrainCo helpers
(`brainco_qpos24_to_cmd12`, `BraincoHandSmoother`,
`load_offline_motions_with_brainco_hands`) are copied verbatim from
`deploy.brainco.play_track_brainco` -- if you fix the conversion math
there, fix it here too.

The terminal UI now reports network health on the `Mocap:` line, e.g.:

```
Mocap: Net  89.6Hz lag= 12.3ms loss=0 ooo=0
```

- `Hz` -- packets/sec actually delivered to the latch.
- `lag` -- wall-clock delay from host `time.time()` at send to receiver
  receive (depends on WiFi quality + clock skew between the two boxes; only
  the trend matters, not the absolute value).
- `loss` -- estimated lost-in-flight packets (gaps in the sender sequence).
- `ooo` -- out-of-order packets we discarded.

### Modes

Identical to `play_track_onboard.py`:

- **0** Walk policy (WASDQE for velocity command).
- **1** Online retarget -- now driven by network instead of local GMR.
- **2..9** Offline trajectories from `--track-dir`.

### 3) (Optional) Verify the network path only -- `bench_net_recv`

Before bringing the full motor loop up, you can sanity-check the WiFi link
and packet integrity in isolation.  This script imports neither MuJoCo nor
the tracking policy nor DDS, so it is safe to run on the G1 with the robot
powered down:

```bash
# On the G1 (while host_sender is running on the workstation):
python -m deploy.onboard_deploy_wo_GMR.bench_net_recv

# With BrainCo packets:
python -m deploy.onboard_deploy_wo_GMR.bench_net_recv --has-brainco

# Or, fully local (no workstation, no Noitom) -- spawn a synthetic sender
# on loopback to verify the receiver code path itself:
python -m deploy.onboard_deploy_wo_GMR.bench_net_recv \
    --self-test --listen-ip 127.0.0.1 --duration-sec 5

# Self-test with synthetic BrainCo qpos too:
python -m deploy.onboard_deploy_wo_GMR.bench_net_recv \
    --self-test --self-test-brainco --has-brainco \
    --listen-ip 127.0.0.1 --duration-sec 5
```

The summary reports throughput, latency percentiles, packet integrity, and
content sanity:

```
  expected pkt     : 276 B  (has_hand=True, has_brainco=True)
  received         : 2700 (89.97 Hz, 194.0 kbps)
  bad packets      : 0 (wrong magic / length / dof)
  out-of-order     : 0 (older send_ts -> dropped)
  estimated loss   : 0 (0.00% of in-flight)
  arrival dt (snd) : mean=  11.111ms  p50=  11.083ms  p90=  12.683ms ...
  net lag (snd->rd): mean=   3.142ms  p50=   3.103ms  p90=   5.247ms ...
  root |q| sanity  : mean=1.000000  max|err|=0.0000  off>1e-2:0/2700
  root height (m)  : mean=0.750  min=0.750  max=0.750
  brainco_qpos     : seen=2700 frames  min(mean)=+0.000  max(mean)=+0.500  nan_frames=0
```

What to look for in a healthy WiFi run against the real workstation:
- `received Hz` is close to the workstation's send Hz (~90 by default).
- `bad packets` stays at 0 (any non-zero count means `encode/decode_frame`
  mismatch -- e.g. a version skew between the two boxes).
- `estimated loss` < 1% is good, < 0.1% is excellent.
- `root |q| sanity` shows `off>1e-2: 0/N` -- the root quaternion is always
  unit-norm, meaning the float32 layout was unpacked correctly.
- `net lag` is dominated by clock skew between the two boxes unless NTP
  is running on both -- look at the *trend* over a long run, not the
  absolute value.
- (BrainCo only) `brainco_qpos: seen=N frames` should equal `received` and
  `nan_frames=0`.  If you see `NOT received`, the workstation forgot
  `--enable-brainco-hand`.

## Troubleshooting

| Symptom                                                 | Likely cause                                                                 | Fix                                                                                |
|---------------------------------------------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| `Mocap: NO DATA (...)` on G1                            | Wrong `--robot-ip` on workstation, or firewall.                              | `ping` both directions; check `--robot-port == --listen-port`.                     |
| `Net  0.0Hz` even though host shows `send= 89Hz`        | WiFi isolation between workstation and robot (router AP isolation enabled).  | Disable AP isolation, or put both on a dedicated 2.4/5 GHz SSID.                   |
| `lag` keeps growing during a session                    | Workstation clock drift (NTP off).                                           | Enable NTP / chrony on both boxes; the `lag` field is wall-clock based.            |
| `loss` increases steadily                               | Congested or weak WiFi.                                                      | Move the G1 closer to the AP, or drop `--send-hz 60` on the workstation.           |
| Host `dup_polls` ~= GMR output rate                     | Normal -- the workstation polls slightly faster than GMR produces frames.    | No action.                                                                         |
| Robot CPU still saturating in tracking mode             | TRT not enabled on G1, or other process competing for core 4.                | `--use-trt true`; check `htop` for other workloads on the Jetson.                  |
| `[net_recv] RT pin failed (...)` printed at startup     | Process lacks `CAP_SYS_NICE`.                                                | Run with `sudo`, or `setcap cap_sys_nice+ep $(which python)`, or pass `--net-recv-rt-pin None`. |
| Audible motor "click" jitter while tracking             | Running an old build of `net_recv.py` that used a same-process thread.       | `git pull` -- the receiver is now a SCHED_FIFO subprocess. Verify with `pidstat -p $(pgrep -f netmocap-rx) 1`. |

## When to use which mode

- Bench testing / motion debugging at the workstation: **`play_track.py`** (full GUI, low latency over Ethernet).
- BrainCo dex-hand testing at the workstation: **`deploy/brainco/play_track_brainco.py`**.
- Field demo, no laptop tethered, willing to spend Jetson CPU: **`onboard_deploy/play_track_onboard.py`** (fully self-contained on the robot).
- Field demo with a 4090 nearby on the same WiFi, want lowest robot-side CPU and no cable, **Dex3 hands**: **`play_track_onboard_wo_GMR.py`**.
- Same as above, but with **BrainCo dex-hands**: **`play_track_onboard_wo_GMR_brainco.py`** (host: `--enable-brainco-hand`).
