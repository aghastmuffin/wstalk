# wstalk / WiGest

WiFi-based gesture recognition system implementing the
[WiGest (INFOCOM 2015)](https://ieeexplore.ieee.org/document/7218622) paper
as a pure-Python library.

> **No GUI, no OS media controls** — just a clean Python API that you wire
> up to your own application.

---

## Overview

WiGest detects hand gestures by analysing the Received Signal Strength
Indicator (RSSI) of nearby WiFi access points.  A hand moving near the
antenna perturbs the multipath propagation, creating characteristic
patterns in the RSSI time series that the library classifies into gesture
families.

Full pipeline:

```
WiFi RSSI  →  DWT denoising  →  Preamble  →  Primitive extraction
           →  Multi-AP voting  →  Gesture matching  →  Action dispatch
```

---

## Package structure

```
wigest/
├── __init__.py      # public API re-exports
├── rssi.py          # RSSI acquisition (Linux/Windows/mock)
├── denoising.py     # DWT/Haar + SURE thresholding
├── primitives.py    # edge extraction + primitive classification
├── preamble.py      # preamble detection + calibration
├── voting.py        # multi-AP majority voting
├── gestures.py      # pattern encoding + gesture family matching
├── actions.py       # application-facing action mapper
└── pipeline.py      # end-to-end WiGest controller

tests/
├── test_denoising.py
├── test_primitives.py
├── test_preamble.py
├── test_gestures.py
├── test_voting.py
└── test_pipeline.py

example.py           # runnable demo script
requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `PyWavelets`.

---

## Quick start

### Simulated (no hardware needed)

```bash
python example.py --simulate --no-preamble
```

### Live WiFi (Linux)

```bash
python example.py --interface wlan0
```

### Live WiFi (Windows)

```bash
python example.py --interface "Wi-Fi"
```

---

## Python API

```python
from wigest import WiGest, MockRSSISource, ActionMapper
import numpy as np

# --- Offline / batch processing ---
signal = np.load("my_rssi.npy")          # shape (N,), dBm
wg = WiGest.from_array(signal, preamble=False)
for gesture in wg.run(max_gestures=10):
    print(gesture)

# --- With action bindings ---
mapper = ActionMapper()

@mapper.on("push")
def volume_up(result):
    print("Volume UP")

@mapper.on("pull")
def volume_down(result):
    print("Volume DOWN")

source = MockRSSISource(signal)
wg = WiGest(source, action_mapper=mapper, preamble=False)
list(wg.run())                           # blocks until source exhausted
```

---

## Gesture families

| Family   | Pattern | Description            |
|----------|---------|------------------------|
| `push`   | `+-`    | Single push            |
| `pull`   | `-+`    | Single pull            |
| `push_n` | `(+-){2,}` | Repeated push       |
| `pull_n` | `(-+){2,}` | Repeated pull       |
| `left`   | `+-0-`  | Push–pause–pull        |
| `right`  | `-+0+`  | Pull–pause–push        |
| `up`     | `+`     | Single rising edge     |
| `down`   | `-`     | Single falling edge    |
| `circle` | varies  | Circular motion        |

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Reference

H. Abdelnasser, M. Youssef, and K. A. Harras,
"WiGest: A Ubiquitous WiFi-Based Gesture Recognition System,"
*IEEE INFOCOM 2015*.
