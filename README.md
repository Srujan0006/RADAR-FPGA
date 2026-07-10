# Hardware Flow Handoff: Custom HLS Radar CNN Accelerator (KV260)

## Project Goal

Replace the Vitis AI DPU inference path with a **custom HLS CNN accelerator** running directly on the KV260 FPGA.

The objective is to be able to state:

> "Inference is performed using a custom FPGA hardware accelerator generated from C++ HLS rather than the Vitis AI DPU."

This is an architectural project rather than simply deploying a CNN onto the DPU.

---

# Current Project Status

The software side of the project is essentially complete.

Completed:

- ✔ Record-wise preprocessing
- ✔ Leakage-free evaluation
- ✔ PyTorch training
- ✔ DPU-friendly CNN architecture
- ✔ ONNX export
- ✔ Vitis AI quantization
- ✔ xmodel generation
- ✔ KV260 compilation
- ✔ Optional DPU deployment path
- ✔ HLS accelerator implemented
- ✔ Host-float validation
- ✔ Fixed-point preparation pipeline

The only remaining large engineering task is making the custom HLS accelerator efficient enough to fit on the KV260.

---

# Current HLS Architecture

The accelerator currently implements the full CNN.

Pipeline:

```
Input Image (100×100)

↓

Stem 3×3 Conv

↓

Depthwise Conv

↓

Pointwise Conv

↓

MaxPool

↓

Depthwise Conv

↓

Pointwise Conv

↓

MaxPool

↓

Depthwise Conv

↓

Pointwise Conv

↓

MaxPool

↓

Depthwise Conv

↓

Pointwise Conv

↓

Global Average Pool

↓

1×1 Classifier

↓

4 logits
```

The arithmetic is correct.

---

# Current Verification Status

## Software Float

Verified.

The software version of the HLS accelerator matches PyTorch almost exactly.

Maximum logit error:

```
≈ 8e-7
```

Predictions match.

---

## C Simulation

Completed successfully.

Real radar samples were used.

Output:

```
Sample 0 True=0 Pred=0
Sample 1 True=0 Pred=0
...
Sample 7 True=3 Pred=3
```

Therefore:

Algorithm is correct.

---

## HLS Synthesis

Successfully synthesized.

Top function:

```
radar_cnn_accel()
```

Clock:

```
5 ns
```

Target:

```
xczu3eg-sbva484-1-e
```

(KV260)

---

# Major Bottleneck

The accelerator **does not fit** on the FPGA.

Not because of DSP.

Not because of LUTs.

Almost entirely because of memory architecture.

---

## Original synthesis

Resources:

```
URAM ≈221

BRAM ≈777

DSP ≈817

LUT ≈75k
```

Problem:

The intermediate feature maps were explicitly forced into URAM using

```
#pragma HLS BIND_STORAGE
```

Removing those pragmas eliminated URAM usage.

---

## After removing URAM pragmas

Resources became approximately

```
URAM = 0

BRAM ≈1600

DSP ≈817

LUT ≈76k
```

This proved something extremely important.

The issue is NOT URAM.

The issue is storing every intermediate feature map.

Vitis simply moved them into BRAM.

---

# Root Cause

Current accelerator stores complete tensors.

Example:

```
pw1

32

×

100

×

100
```

Entire tensor stored.

Then read.

Then discarded.

Every stage behaves this way.

Examples:

```
stem

↓

store entire output

↓

dw1

↓

store entire output

↓

pw1

↓

store entire output

↓

pool

↓

...
```

This architecture behaves like software.

It is not how FPGA CNN accelerators are normally built.

---

# What Was Tried

## Reduced unrolling

Partially successful.

Reduced URAM slightly.

Did not solve BRAM explosion.

---

## Removed BIND_STORAGE pragmas

Successful.

URAM became zero.

BRAM increased dramatically.

Confirmed storage architecture is the bottleneck.

---

## AP_SAT

Changed

```
AP_WRAP

↓

AP_SAT
```

Kept.

Correct decision.

---

## AXI widening

Attempted

```
config_interface -m_axi_max_widen_bitwidth 512
```

Did not actually take effect.

Still reports

```
threshold = 0
```

Needs revisiting later.

Not the primary bottleneck.

---

# Important Engineering Conclusion

The current architecture has reached its limit.

Further pragma tuning will not reduce

```
1600 BRAM

↓

<300 BRAM
```

The architecture itself must change.

---

# New Architecture (Version 2)

Implement a streaming CNN.

Instead of

```
Feature Map

↓

Store

↓

Read

↓

Store

↓

Read
```

Use

```
Stage

↓

FIFO

↓

Stage

↓

FIFO

↓

Stage
```

using

```
hls::stream<>
```

---

# Streaming Concept

Current

```
Input

↓

Stem

↓

100×100×16 buffer

↓

Depthwise

↓

100×100×16 buffer

↓

Pointwise

↓

100×100×32 buffer
```

New

```
Input

↓

Stem

↓

FIFO

↓

Depthwise

↓

FIFO

↓

Pointwise

↓

FIFO
```

No complete feature maps.

Only small FIFOs.

---

# Layer-by-layer Changes

## Stem

Current

Stores

```
16×100×100
```

New

Uses

```
3-row line buffer
```

---

## Depthwise

Current

Stores entire tensor.

New

Needs only

```
3 rows
```

because

```
3×3 kernel
```

Only local neighbourhood required.

---

## Pointwise

This is the biggest optimization.

A

```
1×1
```

convolution has

no spatial neighbourhood.

It only needs

```
all channels

for one pixel.
```

Therefore

Instead of storing

```
100×100×32
```

it processes

```
pixel vector

↓

output vector

↓

next stage
```

Nearly zero storage.

This is expected to eliminate the largest BRAM users.

---

## MaxPool

Needs only

```
2 rows
```

Again

No full feature map.

---

## Global Average Pool

Can remain mostly unchanged.

Very small memory.

---

# DATAFLOW

Eventually the top function becomes

```
#pragma HLS DATAFLOW

Read

↓

Stem

↓

DW1

↓

PW1

↓

Pool

↓

DW2

↓

PW2

↓

Pool

↓

DW3

↓

PW3

↓

Pool

↓

DW4

↓

PW4

↓

GAP

↓

Classifier

↓

Write
```

All stages execute simultaneously.

---

# Development Strategy

Do NOT rewrite everything at once.

Incremental approach.

---

## Stage 1

Streaming input

↓

Stem

Verify.

---

## Stage 2

Stem

↓

DW1

Verify.

---

## Stage 3

Add PW1.

Verify.

---

## Stage 4

Add Pool.

Verify.

---

Continue until the entire CNN is streaming.

---

# Expected Benefits

Current

```
Huge memories

↓

Very high BRAM

↓

Cannot fit
```

Streaming

```
Tiny line buffers

↓

Tiny FIFOs

↓

Fits FPGA
```

---

Expected improvements

Massive BRAM reduction

Lower latency

Higher throughput

True FPGA pipeline

---

# Things NOT To Change

Keep

- Trained weights
- CNN topology
- Number of layers
- Dataset
- Spectrogram generation
- Fixed-point types
- AP_SAT
- Existing validation scripts

Only the hardware architecture changes.

---

# Repository Status

Verified working

```
hardware/hls/radar_cnn_hls_accel/
```

Important files

```
radar_cnn_hls.cpp
radar_cnn_hls.h
weights.h
tb_real_samples.cpp
run_hls.tcl
```

Validation

```
compare_fixed_point.py
```

Reports

```
reports/fixed_point_validation/
```

---

# Remaining Milestones

## Immediate

- Rewrite HLS accelerator into a streaming architecture using `hls::stream<>`.
- Replace full-frame feature maps with 3-row (conv) and 2-row (pool) line buffers.
- Add `#pragma HLS DATAFLOW` progressively, verifying after each stage.

## After Streaming Works

- Re-run C simulation.
- Re-run C synthesis.
- Compare BRAM/URAM/DSP/LUT usage against the baseline.
- Check timing (5 ns target).
- Verify numerical equivalence with PyTorch using the existing validation scripts.

## Final

- Export RTL/IP.
- Integrate into a Vivado block design on the KV260.
- Run inference on hardware.
- Benchmark latency, throughput, FPS, and compare against the CPU and Vitis AI DPU paths.

---

# Final Engineering Insight

The current accelerator is **functionally correct but memory-bound**. All algorithmic verification is complete (PyTorch → HLS → C simulation). The remaining work is **architectural optimization**. The correct path forward is not more pragma tuning but a transition from a **frame-buffered CNN** to a **streaming DATAFLOW accelerator**, which is the standard architecture used in published FPGA CNN accelerators and offers the best chance of fitting on the KV260 while providing a compelling custom hardware implementation.
