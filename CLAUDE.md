# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This repo wraps the workflow for training/exporting a custom YOLOv8n model and deploying it to the **Himax WE2** edge device (Arm Cortex-M55 + Ethos-U55 NPU). The two top-level Python scripts are the active work; `YOLOv8_on_WE2/` is a vendored copy of the upstream Himax reference repo, used as the deployment toolkit.

## Environment

- Local virtualenv lives in `venv/` (Windows). Activate with `source venv/Scripts/activate`.
- Dependencies: `pip install -r requirements.txt` (pins `ultralytics==8.4.36`, `torch==2.11.0`, `tensorflow==2.19.0`, `onnx2tf==1.28.8`, `ethos-u-vela==5.0.0`, `opencv-python`).
- `.gitignore` excludes `*.pt`, `*.onnx`, `*.tflite`, `*.mp4`, and venvs — model artifacts and media must not be committed.

## Common commands

- **Export PyTorch → int8 TFLite** (active script): `python pt_to_tflite.py`. Edit the `image_size`, source `.pt`, and `data=` yaml inside the script. `data` must point at a yaml matching the model's actual classes (currently set to `coco128.yaml` as a placeholder — change before exporting a custom-class model, otherwise quantization calibration uses the wrong dataset).
- **Visualize on a video**: `python yoloModelVisual.py`. Edit `SOURCE`, `MODEL_PATH`, `OUTPUT`, `CONF_THRESH`, `IOU_THRESH` at the top of the file.
- **Vela compile** (after TFLite export, run from `YOLOv8_on_WE2/vela/`):
  ```
  vela --accelerator-config ethos-u55-64 --config himax_vela.ini \
       --system-config My_Sys_Cfg --memory-mode My_Mem_Mode_Parent \
       --output-dir <out> <input>.tflite
  ```

There is no test suite, lint config, or build script at the top level.

## Architecture / pipeline

The pipeline is sequential and each stage's output feeds the next:

1. **Train** (Ultralytics Hub or local) → `best.pt`.
2. **Export to int8 TFLite** via `model.export(format="tflite", int8=True, imgsz=...)`. **`imgsz` must be 192** (or 256 for pose) — this is a hard Himax WE2 constraint, not a tunable. Internally Ultralytics goes `.pt → ONNX → onnx2tf → TFLite` and uses `data` for INT8 calibration.
3. **(Optional) Strip transpose ops** before the TFLite step. Vela cannot offload `transpose`, so they fall back to the M55 CPU and dominate latency. The fix is to pass `-prf <param_replacement.json> -rtpo` into the `onnx2tf` call inside Ultralytics' exporter (see `YOLOv8_on_WE2/replace_192_80cls_transpose_op.json`). If the model's class count is not 80, the constant `144` in that JSON must be replaced with `64 + class_num`.
4. **Vela compile** the TFLite into an Ethos-U55-optimized TFLite. Aim for `Total SRAM used < 1 MB` in the report.
5. **Build & run on FVP** (Corstone SSE-300, Cortex-M55 + Ethos-U55) via the `ml-embedded-evaluation-kit` flow documented in `YOLOv8_on_WE2/README.md`. **This stage requires Ubuntu 20.04 LTS** and the GNU Arm Embedded Toolchain `10-2020-q4` — it does not run on Windows. Steps 1–4 do run on Windows.

## Critical gotchas

- **MAC count must match end-to-end.** Vela is invoked with `--accelerator-config ethos-u55-64` and the FVP must be launched with `-C ethosu.num_macs=64`. A mismatch causes runtime invoke failure, not a build error.
- **Calibration dataset matters.** The `data=` argument to `model.export(...)` selects the calibration set for INT8 quantization. Wrong dataset → silent accuracy regression after quantization, not a crash.
- **`YOLOv8_on_WE2/` is vendored, not a submodule.** Treat it as part of this repo, but its build instructions and Python deps (e.g. `tensorflow==2.13.1`, `onnx2tf==1.15.4` for the DeGirum pose path) intentionally diverge from the top-level `requirements.txt`. Use a separate venv if running the pose export flow.
- A stray file named `=4.10.0` exists at the repo root — it is captured pip stdout from a malformed install command, not a real artifact. Safe to delete.
