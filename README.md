# Traffic Sign Recognition with Neural Compute Stick

> **Course project** — NIE-EHW Embedded Systems, [Czech Technical University in Prague](https://www.cvut.cz/en) (CVUT), 2025

Real-time traffic sign detection on the [LIVS](https://dcgi.fel.cvut.cz/home/cech/livs/) self-driving car platform using **YOLOv3-tiny**, **Intel Neural Compute Stick 2** and a **Raspberry Pi 4**.

<p align="center">
  <img src="docs/intro.jpeg" width="520" alt="LIVS car platform"/>
</p>

## Highlights

| | |
|---|---|
| **Model** | YOLOv3-tiny (custom *tinyu* variant, Ultralytics) |
| **Accuracy** | mAP50 **0.87** &mdash; Precision 0.95 &mdash; Recall 0.82 |
| **Inference** | ~180 ms on NCS2 (MYRIAD), ~500 ms end-to-end |
| **Classes** | 15 &mdash; traffic lights, speed limits (10-120), stop |
| **Streaming** | Low-latency UDP + H.264 via FFmpeg |

## Architecture

```
USB Camera ──► Raspberry Pi 4 ──► Intel NCS 2    Arduino Nano
  320x240        Python 3.10       (inference)    (vehicle ctrl)
                      │
                      ▼ UDP / H.264
                  Client PC
                  (ffplay)
```

**Pipeline:** capture &rarr; letterbox resize (352x352) &rarr; OpenVINO inference &rarr; NMS post-processing &rarr; annotate &rarr; H.264 encode &rarr; UDP stream.

## Project Structure

```
src/inference.py            Inference + streaming script (Raspberry Pi)
notebooks/training.ipynb    Training notebook (Jupyter)
models/
  yolov3-tinyu.pt           Pre-trained base model
  best.onnx                 Trained model (ONNX)
  openvino_ir/              OpenVINO IR for NCS2 (.xml/.bin)
training_results/           Metrics, curves and confusion matrices
requirements/
  training.txt              GPU workstation dependencies
  inference.txt             Raspberry Pi dependencies
docs/                       Presentation (Reveal.js) and task report
```

## Quick Start

### Training (GPU workstation)

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements/training.txt
```

Open `notebooks/training.ipynb`, set `path_to_repository`, and run all cells.
The dataset is **not included** — download it from [Roboflow](https://universe.roboflow.com/selfdriving-car-qtywx/self-driving-cars-lfjou/dataset/6) and place it under `dataset/`.

### Inference (Raspberry Pi + NCS2)

```bash
python3.10 -m venv ov2022 && source ov2022/bin/activate
pip install -r requirements/inference.txt

# Verify NCS2 is detected
python -c "from openvino.inference_engine import IECore; print(IECore().available_devices)"
# Expected: ['CPU', 'MYRIAD']

python src/inference.py
```

On the client PC, open the stream:

```bash
ffplay -fflags nobuffer -flags low_delay -framedrop udp://@:5001
```

### Model Export (ONNX &rarr; OpenVINO)

```bash
mo --input_model models/best.onnx \
   --output_dir models/openvino_ir/ \
   --input_shape [1,3,352,352] \
   --data_type FP16
```

## Performance

| Stage | Latency |
|-------|---------|
| Camera capture | ~30 ms |
| Preprocessing | ~15 ms |
| **Inference (NCS2)** | **~180 ms** |
| Post-processing + drawing | ~170 ms |
| H.264 encoding | ~40 ms |
| Network (UDP) + decoding | ~65 ms |
| **End-to-end** | **~500 ms** |

### Training Results (100 epochs)

| Metric | Value |
|--------|-------|
| mAP50 | 0.871 |
| mAP50-95 | 0.773 |
| Precision | 0.954 |
| Recall | 0.816 |

## Important Notes

- **Intel NCS2 was discontinued in 2022.** Only OpenVINO **2022.3** supports the MYRIAD device. Newer versions will *not* work.
- The `inference.py` script handles the full YOLOv3-tiny post-processing (box decoding, NMS) manually, because the OpenVINO 2022.3 legacy API does not provide it.
- USB udev rules may be needed for NCS2 access — see [Troubleshooting](#troubleshooting) below.

## Troubleshooting

<details>
<summary><strong>NCS2 not detected</strong></summary>

```bash
# Check USB connection
lsusb | grep Movidius

# Install udev rules
sudo usermod -a -G users $USER
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/97-myriad.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Make sure you're using `openvino==2022.3.0`.
</details>

<details>
<summary><strong>Camera not opening</strong></summary>

```bash
ls /dev/video*          # check device exists
ffplay /dev/video0      # test capture
sudo apt install v4l-utils
```
</details>

<details>
<summary><strong>H.264 encoder not found</strong></summary>

Replace `h264_v4l2m2m` with `libx264` in `inference.py` to fall back to software encoding.
</details>

## Authors

| Name | Contribution |
|------|:---:|
| **Tom Mafille** | 50% |
| **Lukas Prendky** | 50% |

## License

[MIT](LICENSE) &mdash; Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## References

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [OpenVINO 2022.3](https://docs.openvino.ai/2022.3/)
- [Dataset (Roboflow)](https://universe.roboflow.com/selfdriving-car-qtywx/self-driving-cars-lfjou/dataset/6)
