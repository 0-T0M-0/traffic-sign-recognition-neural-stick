#!/usr/bin/env python3
"""
Ultra Low Latency Detection with YOLOv8
Optimized for minimal latency with ffplay
"""

import cv2
import numpy as np
from openvino.inference_engine import IECore
import subprocess
import time
import glob

# CONFIGURATION
MODEL_XML = "models/openvino_ir/best.xml"
MODEL_BIN = "models/openvino_ir/best.bin"
PC_IP = "172.20.10.2"
PC_PORT = 5001

# Detection parameters
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# LATENCY OPTIMIZATION
RESOLUTION = (320, 240)  # Reduced from 640x480 to 320x240
FRAMERATE = 15           # Increased from 10 to 15 FPS
BITRATE = "1.5M"         # Reduced from 3M to 1.5M

# Load class names if available
try:
    from model_classes import CLASS_NAMES
    CLASSES = CLASS_NAMES
except:
    CLASSES = []

# HELPER FUNCTIONS
def find_camera_device():
    """Find first working camera device"""
    video_devices = glob.glob('/dev/video*')
    if not video_devices:
        return None
    for device in sorted(video_devices):
        cap = cv2.VideoCapture(device)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                return device
            cap.release()
    return None

def open_camera_with_retry(device_path, resolution, max_retries=5, retry_delay=2):
    """Open camera with retry mechanism and minimal buffer"""
    for attempt in range(max_retries):
        cap = cv2.VideoCapture(device_path)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # CRITICAL: minimal buffer
            time.sleep(1)
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            cap.release()
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    return None

def letterbox_resize(img, target_size):
    """Resize image preserving aspect ratio with letterboxing"""
    h, w = img.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return padded, scale, (pad_left, pad_top)

def xywh2xyxy(boxes):
    """Convert boxes from xywh (center) to xyxy (corners) format"""
    boxes_xyxy = np.copy(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return boxes_xyxy

def nms(boxes, scores, iou_threshold):
    """Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def postprocess_yolov8(output, img_shape, input_shape, scale, pad_offset, conf_threshold, iou_threshold):
    """
    Post-processing for YOLOv8 modern format
    Expected format: [1, num_predictions, 4+num_classes]
    """
    # Extract batch
    predictions = output[0]

    # Check and transpose if necessary
    if len(predictions.shape) == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]

    # Format: [num_predictions, data] or [data, num_predictions]
    if predictions.shape[0] < predictions.shape[1]:
        # Transposed format [data, num_predictions] -> [num_predictions, data]
        predictions = predictions.T

    # Now: [num_predictions, 4+num_classes]
    # Extract boxes and scores
    boxes = predictions[:, :4]  # x_center, y_center, w, h
    scores_classes = predictions[:, 4:]  # scores for each class

    # Get class and score
    class_ids = np.argmax(scores_classes, axis=1)
    scores = np.max(scores_classes, axis=1)

    # Filter by confidence
    mask = scores > conf_threshold
    boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Convert xywh -> xyxy
    boxes_xyxy = xywh2xyxy(boxes)

    # Remove padding and rescale
    pad_left, pad_top = pad_offset
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_left) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_top) / scale

    # Clip to image dimensions
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, img_shape[1])
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, img_shape[0])

    # Apply NMS
    keep = nms(boxes_xyxy, scores, iou_threshold)

    return boxes_xyxy[keep], scores[keep], class_ids[keep]

# MAIN
print("\n" + "="*60)
print("  ULTRA LOW LATENCY DETECTION")
print("="*60)

# Load model
print("\nLoading OpenVINO model...")
ie = IECore()
net = ie.read_network(model=MODEL_XML, weights=MODEL_BIN)

# Choose device
available_devices = ie.available_devices
print(f"Available devices: {available_devices}")

if "MYRIAD" in available_devices:
    device = "MYRIAD"
    print("Using Neural Compute Stick (MYRIAD)")
else:
    device = "CPU"
    print("Using CPU")

exec_net = ie.load_network(network=net, device_name=device)

input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))
input_shape = net.input_info[input_blob].input_data.shape
output_shape = net.outputs[output_blob].shape

print(f"Model loaded successfully")
print(f"Input shape: {input_shape}")
print(f"Output shape: {output_shape}")

# Detect number of classes
if len(output_shape) >= 2:
    # Modern format: [1, num_pred, 4+classes] or [1, 4+classes, num_pred]
    data_size = min(output_shape[1], output_shape[2]) if len(output_shape) == 3 else output_shape[1]
    if data_size > 4:
        num_classes = data_size - 4
        print(f"Detected {num_classes} classes")
        if not CLASSES:
            CLASSES = [f"Class {i}" for i in range(num_classes)]

time.sleep(1)

# Open camera
print("\nOpening camera...")
camera_device = find_camera_device()
if not camera_device:
    print("ERROR: No camera found")
    exit(1)

cap = open_camera_with_retry(camera_device, RESOLUTION)
if not cap:
    print("ERROR: Cannot open camera")
    exit(1)

print(f"Camera opened: {RESOLUTION[0]}x{RESOLUTION[1]}")

# FFmpeg ULTRA LOW LATENCY configuration
print("\nStarting UDP stream (ultra low latency mode)...")
ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-pix_fmt', 'yuv420p',
    '-s', f'{RESOLUTION[0]}x{RESOLUTION[1]}',
    '-r', str(FRAMERATE),
    '-i', '-',
    '-an',  # No audio

    # LATENCY OPTIMIZATIONS
    '-c:v', 'h264_v4l2m2m',
    '-b:v', BITRATE,
    '-maxrate', BITRATE,
    '-bufsize', '500k',  # Minimal buffer

    # Ultra-fast preset
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',

    # Minimal GOP for latency reduction
    '-g', '10',          # Keyframe every 10 frames (reduced from 15)
    '-keyint_min', '10',
    '-sc_threshold', '0',

    # No B-frames
    '-bf', '0',

    # Minimal threads
    '-threads', '2',

    # Low delay flags
    '-flags', 'low_delay',
    '-fflags', 'nobuffer+fastseek+flush_packets',

    # Output
    '-pix_fmt', 'yuv420p',
    '-f', 'mpegts',
    f'udp://{PC_IP}:{PC_PORT}?pkt_size=1316&buffer_size=65535'
]

proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(0.5)

print("\nStarting detection")
print(f"Resolution: {RESOLUTION[0]}x{RESOLUTION[1]}")
print(f"FPS: {FRAMERATE}")
print(f"Bitrate: {BITRATE}")
print(f"Confidence threshold: {CONF_THRESHOLD}")
print(f"Device: {device}")
print("\nOn PC, run ffplay with:")
print(f"  ffplay -fflags nobuffer -flags low_delay -framedrop udp://{PC_IP}:{PC_PORT}")
print("\nPress Ctrl+C to stop\n")

frame_count = 0
detection_count = 0
last_print_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        # Preprocessing
        target_size = (input_shape[2], input_shape[3])
        preprocessed, scale, pad_offset = letterbox_resize(frame, target_size)
        blob = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = blob.transpose((2, 0, 1))[np.newaxis, :, :, :]

        # Inference
        result = exec_net.infer(inputs={input_blob: blob})
        output = result[output_blob]

        # Post-processing
        boxes, scores, class_ids = postprocess_yolov8(
            output, frame.shape, target_size, scale, pad_offset,
            CONF_THRESHOLD, IOU_THRESHOLD
        )

        detection_count += len(boxes)

        # Draw annotations (lightweight)
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)

            label = CLASSES[class_id] if class_id < len(CLASSES) else f"C{class_id}"

            # Color by class
            if 'stop' in label.lower():
                color = (0, 0, 255)  # Red for STOP
            elif 'light' in label.lower():
                color = (0, 255, 255)  # Yellow for traffic lights
            else:
                color = (0, 255, 0)  # Green default

            # Thin lines for performance
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, f"{label[:5]}:{score:.1f}", (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Logging
        current_time = time.time()
        if frame_count % 100 == 0 or (current_time - last_print_time) >= 10:
            elapsed = current_time - last_print_time
            fps = 100 / elapsed if frame_count % 100 == 0 else 0
            avg_det = detection_count / frame_count
            log_msg = f"Frame {frame_count}: {len(boxes)} det | Avg: {avg_det:.2f}/f"
            if fps > 0:
                log_msg += f" | {fps:.1f} FPS"
            print(log_msg)
            last_print_time = current_time

        # Streaming
        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        try:
            proc.stdin.write(frame_yuv.tobytes())
            proc.stdin.flush()
        except BrokenPipeError:
            break

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    cap.release()
    proc.stdin.close()
    proc.terminate()
    avg_det = detection_count / frame_count if frame_count > 0 else 0
    print(f"\nStats: {frame_count} frames, {detection_count} detections, avg: {avg_det:.2f}/frame")
    print(f"Theoretical latency: ~0.2-0.4s (vs 1-1.5s before)")
