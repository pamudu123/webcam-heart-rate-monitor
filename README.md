# Webcam Heart Rate Monitor

Here we are monitoring heart rate in real-time using a webcam.This is based on an algorithm called **Eulerian Video Magnification** which makes it possible to see the colours of the face change as blood rushes in and out of the head. It is able to detect the pulses and calculates the heart rate in beats per minute (BPM).This method performs well in real-time, providing accurate results and maintaining a good frames-per-second rate, even when using a CPU.

**STEPS**
1. Input: Webcam video feed as the input for heart rate measurement.
2. Preprocessing: Use MediaPipe (CVZone) to detect and localize the face region in the video frames.
3. Spatial Decomposition: Decompose the video frames into multiple spatial frequency bands using a pyramid-based approach.
4. Temporal Filtering: Apply band-pass filtering techniques to isolate the desired frequency range associated with the heartbeat.
5. Magnification: Amplify the subtle temporal variations related to the heartbeat for better visibility.
6. Measurement: Extract the amplified signal and estimate the heart rate in beats per minute (bpm) using appropriate signal processing techniques.
7. Visualize Results: Use CVZone LivePlot to visualize the heart rate estimation results.

                                              ┌───────────────────┐
                                              │       Input       │
                                              │   Webcam Video    │
                                              └───────────────────┘
                                                       │
                                                       v
                                              ┌───────────────────┐
                                              │   Preprocessing   │
                                              │   Face Region     │
                                              │    Detection      │
                                              └───────────────────┘
                                                       │
                                                       v
                                              ┌───────────────────┐
                                              │    Spatial        │
                                              │  Decomposition    │
                                              └───────────────────┘
                                                       │
                                                       v
                                              ┌───────────────────┐
                                              │    Temporal       │
                                              │   Filtering       │
                                              └───────────────────┘
                                                       │
                                                       v
                                              ┌───────────────────┐
                                              │   Magnification   │
                                              └───────────────────┘
                                                       │
                                                       v
                                              ┌───────────────────┐
                                              │    Measurement    │
                                              │   Heart Rate      │
                                              │   Estimation      │
                                              └───────────────────┘
                                                       │
                                                       v
                                              ┌───────────────────┐
                                              │    Visualize      │
                                              │    Results        │
                                              │   CVZone LivePlot │
                                              └───────────────────┘

Demonstration Video
https://www.youtube.com/watch?v=jHzfoeVEXHA&ab_channel=Pamudu123Ranasinghe

Reference
http://people.csail.mit.edu/mrub/evm/


