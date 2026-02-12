# 📊 Validation Scripts 
This folder contains the evaluation scripts used to validate AutoPBR v2 in ECCV-style experiments.

The validation suite evaluates:

* Cross-view stability
* Behavioral differences vs baseline
* Human evaluation
* GPS-based pseudo-building clustering

---

# 🔎 Overview

| Script                      | Purpose                                                               |
| --------------------------- | --------------------------------------------------------------------- |
| `gps_clustering.py`         | Cluster photos into pseudo-buildings using GPS                        |
| `stability_validation.py`   | Cross-view material stability (Jaccard similarity)                    |
| `behavioral_metrics.py`     | Paired behavioral metrics (hallucination, abstention, low-confidence) |
| `human_evaluation_eccv.py`  | Interactive manual evaluation tool                                    |

---

# 1️⃣ GPS Clustering (Cross-View Setup)

📄 Script: `gps_clustering.py` 

Clusters Mapillary photos into pseudo-buildings using DBSCAN on GPS coordinates.

### Input

* Mapillary manifest JSON with GPS coordinates

### Output

* `building_clusters.json`

### Example

```bash
python gps_clustering.py ^
  --photos data_input/bryggen/photos_manifest.json ^
  --output data_output/bryggen/building_clusters.json ^
  --radius 20 ^
  --min-samples 3 ^
  --min-views 3
```

### Parameters

| Flag            | Meaning                           |
| --------------- | --------------------------------- |
| `--radius`      | Clustering radius in meters       |
| `--min-samples` | DBSCAN min cluster size           |
| `--min-views`   | Minimum views per cluster to keep |

---

# 2️⃣ Stability Validation (Cross-View Consistency)

📄 Script: `stability_validation.py` 

Computes cross-view stability per pseudo-building using **Jaccard similarity** over material sets.

### Input

* Structured predictions: `materials_v2_filtered_FULL.json`
* Optional baseline predictions
* `building_clusters.json`

### Example

```bash
python stability_validation.py ^
  --hierarchical materials_v2_filtered_FULL.json ^
  --baseline baseline_full_image.json ^
  --clusters data_output/bryggen/building_clusters.json ^
  --output validation_results/stability.csv
```

### Output

* CSV per-building stability
* JSON summary (means per surface)

### Metric

For each building cluster:

[
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
]

Computed per surface:

* road
* wall
* roof

---

# 3️⃣ Behavioral Metrics (Paired Comparison)

📄 Script: `behavioral_metrics.py` 

Compares hierarchical vs baseline predictions using paired image analysis.

### Evaluates

* Hallucination rate
* Abstention rate
* Low-confidence rate
* Agreement rate
* McNemar test (paired statistical test)

### Example

```bash
python behavioral_metrics.py ^
  --hierarchical materials_v2_filtered_FULL.json ^
  --baseline baseline_full_image.json ^
  --output validation_results/behavioral.json ^
  --vis_road 10 ^
  --vis_wall 10 ^
  --vis_roof 5 ^
  --conf_th 0.3
```

### Key Idea

Visibility is inferred **only from hierarchical coverage_pct**.

Baseline has no masks → cannot estimate visibility independently.

---

# 4️⃣ Human Evaluation (ECCV-style)

📄 Script: `human_evaluation_eccv.py` 

Interactive OpenCV-based annotation tool.

Displays:

* Original image
* Mask outline
* Predicted class
* Coverage %

User labels:

| Key | Meaning   |
| --- | --------- |
| 1   | Correct   |
| 2   | Incorrect |
| 3   | Ambiguous |
| 4   | Skip      |
| q   | Quit      |

---

### Example

```bash
python human_evaluation_eccv.py ^
  --predictions materials_v2_filtered_FULL.json ^
  --output validation_results/human_eval.csv ^
  --scope instances ^
  --surfaces wall roof ^
  --sample 100 ^
  --min-coverage 5
```

---

### Print Summary

```bash
python human_evaluation_eccv.py ^
  --output validation_results/human_eval.csv ^
  --summary
```

---

# 🧠 Recommended ECCV Evaluation Pipeline

1. Cluster photos:

   ```
   gps_clustering.py
   ```

2. Run stability:

   ```
   stability_validation.py
   ```

3. Run behavioral paired test:

   ```
   behavioral_metrics.py
   ```

4. Run human validation:

   ```
   human_evaluation_eccv.py
   ```

---

# 📦 Dependencies

Required:

```bash
pip install numpy scikit-learn opencv-python pillow
```

For stability & metrics:

```bash
pip install scipy
```

---

# 🧪 Output Files

| File                     | Description              |
| ------------------------ | ------------------------ |
| `building_clusters.json` | Pseudo-building clusters |
| `stability.csv`          | Per-building stability   |
| `stability.json`         | Stability summary        |
| `behavioral.json`        | Paired metrics report    |
| `human_eval.csv`         | Human annotations        |
