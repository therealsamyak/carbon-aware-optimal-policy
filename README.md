## Optimal Charge Security Camera

We have a battery-powered security camera that runs local pre-trained image-recognition models. A battery is attached both to the charger, and the camera. The camera is context aware about clean energy information externally. Based on the information, we have a controller that looks at current battery percentage, information about external energy, and user-defined metrics (minimum accuracy, minimum latency, how often we run the model etc.), to decide which image-recognition model is loaded based on all these factors. We want to decide when to charge the battery vs when to use the energy to load a higher model for higher accuracy.

THE GOAL OF THIS CODEBASE IS TO OBTAIN A MODEL WITH GOOD WEIGHTS SO IT CAN BE RAN GENERAL-USE IN ALL SCENARIOS TO OPTIMIZE THIS.

**We have a predefined optimizer function that we will use to optimize the model selection.**

## Model Profiling

The model profiler benchmarks YOLOv10 models for power consumption and performance.

### Power & Latency Benchmarks

Power estimates are stored in `model-profiler/power_profiles.json`. Run it with

```bash
python3 model-profiler/benchmark_power.py
```

Change line 40 of `model-profiler/benchmark_power.py` to modify the number of iterations.

### Accuracy

Accuracy values are also stored in `model-profiler/power_profiles.json`, and are scaled up from the COCO mAP 50-95 values, to more accurately be translated to percentages.

## Citations

Electricity Maps. "United States California LDWP 2024 5 minute Carbon Intensity Data". Electricity Maps, ver. July 2, 2025, https://www.electricitymaps.com.

Electricity Maps. "United States Florida FPL 2024 5 minute Carbon Intensity Data". Electricity Maps, ver. July 2, 2025, https://www.electricitymaps.com.

Electricity Maps. "United States Northwest PSEI 2024 5 minute Carbon Intensity Data". Electricity Maps, ver. July 2, 2025, https://www.electricitymaps.com.

Electricity Maps. "United States New York NYIS 2024 5 minute Carbon Intensity Data". Electricity Maps, ver. July 2, 2025, https://www.electricitymaps.com.
