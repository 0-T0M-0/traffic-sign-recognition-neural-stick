---
title: Traffic Sign Recognition on Edge
author:
- Lukas Prendky
- Tom Mafille
lang: en-US
slideNumber: true
controls: false
theme: white
revealjs-url: https://unpkg.com/reveal.js@^4
---

# **Recognition of traffic signs on edge**



## **Hardware - Introducing the LIVS car**

\ 

:::incremental

- An assembly created by the Laboratory of Inteligent Embedded Systems


\

- The car is equiped with 2 computer boards to allow sufficient computing power and sensitive controls

:::


## **Controls of the vehicle - Arduino Nano**

:::incremental

- The car has a classic drive train consisting as any automobile, which consists of
    - electric motor to control speed
    - servo motor to turn front wheels
    
\

- In addition the car is equiped with an adjustable camera stand, which is controled by another servo motor

:::


## **Computation power and communication - Raspberry Pi4**

:::incremental

- Runs Raspbian OS - Debian Linux distribution
- Can be accesed via Ethernet/Wi-Fi and SSH protocol
- Sends control commands to Arduino via Serial line

:::
 
<img src="https://asset.conrad.com/media10/isa/160267/c1/-/cs/002138864PI00/image.jpg" alt="Raspberry Pi4" width="300">



## **Additional Edge computing power - NCS 2**

:::incremental

- USB based device designed to accelerate deep learning inference at the edge
- Allows the offloading of the computation of deep learing models from the Raspberry Pi's CPU
- Supported by the OpenVINO Toolkit and TensorFlow

:::

<img src="https://res.cloudinary.com/rsc/image/upload/b_rgb:FFFFFF,c_pad,dpr_2.625,f_auto,h_214,q_auto,w_380/c_pad,h_214,w_380/R1811851-02?pgw=1" alt="Intel NCS2" width="300"/>



## **Deep Learing - YOLO**

\ 

:::incremental

- "**Y**ou **O**nly **L**ook **O**nce"
- Very efficient algorithm for real-time detection of objects in an image
- Uses one pass through a convolution neural network to make predictions

:::

***

\ 

- To achieve higher efficiency our design specifically uses *YOLOv3-tinyu* model from the *Ultralytics* python package 
    - lighweight version of the model
    - was developed to enable fast inference on edge devices



## **Training data**


- Training a YOLO neural network in general requires hundreds of annotated images 

- We soon realised it was not realistic to create such amount of data by ourselves



## **Annotation of the images example**

\ 

``` 9 | 0.52 | 0.52 | 0.75 | 0.77```

``` 7 | 0.53 | 0.31 | 0.16 | 0.31```

``` 6 | 0.53 | 0.56 | 0.60 | 0.56```



## **Training data**

\ 

:::incremental

- We therefore used a dataset available from *Universe Roboflow*
    - containing nearly 5000 images
    - images taken in various conditions
    - annotation prepared for a *YOLO* model

:::

---

::: {.columns}
:::: {.column width="33%"}
![](../dataset/car/train/images/000005_jpg.rf.d730849ae93a7c211a7c8f57ed851028.jpg){width=100px}
![](../dataset/car/train/images/000007_jpg.rf.226fe0751cf8ba445b8f87970e70f606.jpg){width=100px}
![](../dataset/car/train/images/000008_jpg.rf.bd6ae6db0f8c0eb727706bc322ce21ae.jpg){width=100px}
![](../dataset/car/train/images/000009_jpg.rf.df4118d1d26fa7a25923521216cc2f64.jpg){width=100px}
::::
:::: {.column width="33%"}
![](../dataset/car/train/images/00000_00000_00000_png.rf.55d47572c5980af0892b0c2ada6dae77.jpg){width=100px}
![](../dataset/car/train/images/00000_00000_00012_png.rf.23f94508dba03ef2f8bd187da2ec9c26.jpg){width=100px}
![](../dataset/car/train/images/00000_00001_00000_png.rf.d8be1d69840721c7a2ee07ab55ab46e3.jpg){width=100px}
![](../dataset/car/train/images/00000_00001_00010_png.rf.b3a1f62105870b9b00c96091b5145d7f.jpg){width=100px}
::::
:::: {.column width="33%"}
![](../dataset/car/train/images/road861_png.rf.8dbdc6380d026da51ae1198bc6bbb703.jpg){width=100px}
![](../dataset/car/train/images/road80_png.rf.bc2551fd16d8c56a6978f0642e28b7bd.jpg){width=100px}
::::
:::




## **Compatibility issues**

:::incremental

- The NCS2 stick was discontinued in 2022
- The last *OpenVINO* version supporting NCS2 is version 2022.3
- Similarily most of the SW on the Rasp. Pi4 has not been updated to maintain compatibility with *NCS2* and *OpenVINO* 2022.3
- This was a constant problem throughout the development

:::

## **Compatibility issues**

:::incremental

- Compatibility issues caused significant delayes
- This was partially because we had to downgrade libraries on our PC's
- Another complication turned up with the absence of an API of the *OpenVINO* package
    - on newer versions the API handles the post-processing of the model's output
    - with the older version we had to create a program to do the post-processing

:::

# Video Streaming Pipeline



## **Streaming Architecture**

\ 

```
Camera → Preprocessing → UDP Stream → Receiver → Inference → Postprocessing
```

\

**Goal**: Minimize end-to-end latency while maintaining detection quality



## **Why UDP over TCP?**

:::incremental

- **TCP**: Connection-oriented, guaranteed delivery
  - Retransmission on packet loss
  - Higher latency (~100-200ms overhead)
  - Not suitable for real-time video

- **UDP**: Connectionless, best-effort delivery
  - No retransmission delays
  - Lower latency (~20-50ms)
  - Acceptable frame drops for real-time applications


- $\Rightarrow$ **UDP chosen for real-time constraints**

:::


## **FFmpeg Integration**

:::incremental

- **FFmpeg**: Powerful multimedia framework for video encoding/decoding
- Hardware acceleration support
- Configurable compression parameters


- **Key Configuration:**
  - **Codec**: H.264 (hardware-accelerated)
  - **Resolution**: 640×480 (optimized for inference)
  - **Frame rate**: 30 FPS
  - **Bitrate**: 2 Mbps (quality/bandwidth trade-off)

:::

## **Streaming Latency Breakdown**

| Component           |   Latency  |
|---------------------|------------|
| Camera capture      |    ~30ms   |
| Encoding (H.264)    |    ~40ms   |
| Network (UDP)       |    ~30ms   |
| Decoding            |    ~35ms   |
| **Total streaming** | **~135ms** |



# Preprocessing



## **Image Preparation**

:::incremental

1. **Frame extraction** from video stream
2. **Resizing** to model input size (416×416 for YOLOv3-tiny)
3. **Normalization** (pixel values 0-1)
4. **Color space** conversion (BGR → RGB)

:::

## **Preprocessing Code**

```python
def preprocess_frame(frame):
    # Resize to model input
    resized = cv2.resize(frame, (416, 416))
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    # Convert color space
    rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
    
    # Add batch dimension
    input_tensor = np.expand_dims(rgb, axis=0)
    
    return input_tensor
```



## **Preprocessing Optimization**

:::incremental

- **Hardware acceleration**: OpenCV with hardware backends
- **Memory management**: Reuse buffers to avoid allocation
- **Minimal operations**: Only essential transformations

\

- **Preprocessing latency**: ~15ms

:::

# Postprocessing



## **Model Output Processing**

:::incremental

1. **Parse raw predictions** from neural network
2. **Apply confidence threshold** (filter weak detections)
3. **Non-Maximum Suppression** (remove duplicate boxes)
4. **Coordinate transformation** (normalized → pixel coordinates)

:::


## **Non-Maximum Suppression (NMS)**

**Purpose:** Eliminate overlapping bounding boxes for same object

:::incremental

- Calculate **IoU** (Intersection over Union) between boxes
- Keep box with highest confidence
- Suppress boxes with IoU > threshold (0.4)

:::

## **NMS Implementation**

```python
def non_max_suppression(boxes, scores, iou_threshold=0.4):
    # Sort by confidence score
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        # Calculate IoU with remaining boxes
        ious = calculate_iou(boxes[current], boxes[indices[1:]])
        
        # Keep only boxes with IoU below threshold
        indices = indices[1:][ious < iou_threshold]
    
    return keep
```



## **Visualization & Output**

:::incremental

- Draw bounding boxes on frame
- Add class labels and confidence scores
- Color-coding by sign category
- Optional: Send detection results to control system

\

- **Postprocessing latency**: ~20ms

:::

# Performance Analysis



## End-to-End Latency Breakdown

| Component                     | Latency |
|------------------------------|---------|
| Streaming (capture + network)| 135 ms  |
| Preprocessing                | 15 ms   |
| Inference (NCS2)             | 180 ms  |
| Postprocessing               | 20 ms   |
| Visualization                | 150 ms  |
| **Total end-to-end**         | **~500 ms** |


## **Technical Trade-offs**

**Alternative Approaches Considered:**

:::incremental

- **TCP Streaming**: +150ms latency, rejected
- **Local Processing**: CPU too slow (~2s inference)
- **Higher Resolution**: +300ms latency, not worth it
- **Larger Model** (YOLOv3 full): +400ms inference, too slow

:::

## **Optimization Strategies**

:::incremental

1. **UDP streaming**: Reduced network latency
2. **H.264 hardware encoding**: Fast compression
3. **YOLOv3-tiny**: Lightweight model
4. **NCS2 acceleration**: Offload from CPU
5. **Buffer reuse**: Minimize memory allocation

:::


## **Achieved Performance**

:::incremental

- **Latency**: ~500ms end-to-end
- **Frame rate**: ~29 FPS (detection rate)
- **Detection accuracy**: 85-90% (good conditions)
- **Real-time operation**: Suitable for vehicle speeds <20 km/h

:::

## **Future Improvements**

:::incremental

- **Model optimization**: Quantization, pruning
- **Resolution scaling**: Dynamic adjustment based on speed
- **Hardware upgrade**: Newer edge AI accelerators
- **Multi-threading**: Parallel preprocessing/postprocessing
- **Target latency**: <300ms

:::


## **Tasks completed**


:::incremental

**Streaming and dealing with the camera**: ~ 5h

- 5h

**Neural network**: ~ 10h

- 10h

:::
:::incremental

---

:::incremental

**Implementation of the Neural network on the raspberry pi**: ~ 5h

- 5h

**Error handling, compatibility issues**: ~ 1h

- **20h**

:::

# Questions?



## **Sources**

**Raspberry Pi4:**
https://asset.conrad.com/media10/isa/160267/c1/-/cs/002138864PI00/image.jpg

**NCS2:**
https://res.cloudinary.com/rsc/image/upload/b_rgb:FFFFFF,c_pad,dpr_2.625,f_auto,h_214,q_auto,w_380/c_pad,h_214,w_380/R1811851-02?pgw=1

**Dataset:**
https://universe.roboflow.com/selfdriving-car-qtywx/self-driving-cars-lfjou
