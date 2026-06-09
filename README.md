# Camera Recognition System

### Multi-Camera Identity Recognition System

A modular computer vision platform that processes multiple camera streams concurrently, performs motion, person, and face recognition, and generates identity-aware recognition events in real time.

Designed using a **Spec-Driven Development** approach with a focus on **real-time processing**, **scalability**, **system architecture**.

---

## Demo

<p align="center">
  <img src="assets/demo.gif" width="250">
</p>

<p align="center">
  <i>Real-time visual replay of the complete recognition pipeline</i>
</p>

---

## Architecture

```mermaid
flowchart LR

    subgraph CAM["Camera Sources"]
        C1["Camera 1"]
        C2["Camera 2"]
        C3["Camera N"]
    end

    subgraph GATEWAY["Frame Ingestion Gateway"]
        T1["Receiver Thread 1<br/>Receive + Validate"]
        T2["Receiver Thread 2<br/>Receive + Validate"]
        T3["Receiver Thread N<br/>Receive + Validate"]
    end

    subgraph IPS["Image Processing Service"]

        Q1["Queue 1"]
        Q2["Queue 2"]
        Q3["Queue N"]

        W1["Worker Thread 1"]
        W2["Worker Thread 2"]
        W3["Worker Thread N"]

        subgraph RPM["Recognition Pipeline Manager"]

            MD["Motion Detection"]

            OD["Object Detection"]

            FD["Face Detection"]

            FR["Face Recognition"]

        end
    end

    OUT["Recognition Events"]

    C1 --> T1 --> Q1 --> W1
    C2 --> T2 --> Q2 --> W2
    C3 --> T3 --> Q3 --> W3

    W1 --> RPM
    W2 --> RPM
    W3 --> RPM

    MD --> OD --> FD --> FR --> OUT

    %% ---------- Colors ----------

    style CAM fill:#F5F3FF,stroke:#A78BFA,stroke-width:2px
    style GATEWAY fill:#F8FAFC,stroke:#94A3B8,stroke-width:2px
    style IPS fill:#F8FAFC,stroke:#94A3B8,stroke-width:2px

    style RPM fill:#F5F3FF,stroke:#8B5CF6,stroke-width:2px

    style OUT fill:#ECFDF5,stroke:#22C55E,stroke-width:3px

    classDef queue fill:#FEFCE8,stroke:#CA8A04,stroke-width:1.5px
    classDef worker fill:#EFF6FF,stroke:#3B82F6,stroke-width:1.5px
    classDef thread fill:#FDF4FF,stroke:#C026D3,stroke-width:1.5px

    class T1,T2,T3 thread
    class Q1,Q2,Q3 queue
    class W1,W2,W3 worker
```

---

## Recognition Pipeline

```mermaid
flowchart LR

    Frame["Frame"]
    Motion["Motion Detection"]
    Person["Person Detection"]
    Face["Face Detection"]
    Recognition["Face Recognition"]
    Identity["Identity Matching"]

    Frame --> Motion
    Motion --> Person
    Person --> Face
    Face --> Recognition
    Recognition --> Identity
```

---

## Engineering Highlights

* Multi-Camera Processing
* Per-Camera Isolation
* Real-Time First Design
* Multi-Threaded Architecture
* Bounded Queue Architecture
* Queue-Based Backpressure Management
* Motion-Gated Processing
* Runtime Metrics & Diagnostics
* Replay Framework
* Config-Driven Runtime Behavior
* Spec-Driven Development
* 500+ Automated Tests

---

## Technologies

**Python • OpenCV • YOLO • ONNX Runtime • NumPy • Docker • Linux • Pytest • YAML • Git**

---

## Design Documentation

Detailed design specifications covering system architecture, runtime processing flow, recognition pipeline orchestration, and shared contracts.

* [Image Processing Service](doc/image_processing_service/image_processing_service.md)
* [Recognition Pipeline Manager](doc/image_processing_service/RecognitionPipelineManager.md)
* [Frame Ingestion Gateway](doc/image_processing_service/frame_ingestion_gateway.md)


## Author

**Tali Fruman**
B.Sc. Information Systems (AI Specialization)
University of Haifa
