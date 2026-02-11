# HybridGT: 3D City Generation Pipeline

HybridGT (Hybrid Geotypical/Geospecific) is a technical pipeline designed to transform 2D building footprints (GeoJSON/Shapefile/GeoPackage) into optimized 3D city scenes. It balances performance and visual fidelity by using a **hybrid approach**: common buildings are represented by optimized geotypical prototypes (instanced), while unique or complex buildings are generated as unique geospecific models.

---

## 🛠 Project Structure

- `src/`: Core Python source code.
- `textures/`: Default facade and roof textures.
- `models/`: Pre-trained AI models for dimension prediction.
- `pipeline_output/`: Default directory for generated assets and KPIs.

---

## 🚀 The 7-Step Pipeline Process

The `src/run_pipeline.py` script orchestrates the following technical stages:

### Step 0.5: Polygon Merging (Preprocessing)
Handles fragmented data by merging contiguous or overlapping polygons.
- **Robustness**: Uses spatial joins to preserve attributes (like building levels and height) from the original data, taking the maximum value among merged parts.
- **Topology**: Cleans small holes and ensures a solid manifold footprint.
- **Count Consistency**: Ensures the final polygon count exactly matches the building units processed in later steps.

### Step 1: Dimension Estimation (Predict vs. MOBB)
Calculates the width, height, and orientation of each building:
- **Predict Mode**: Uses a Random Forest model to estimate dimensions from metadata.
- **MOBB Mode (`--use_mobb`)**: Calculates the Minimum Oriented Bounding Box for precise dimensions.
- **Orientation**: Regardless of mode, the final orientation is always calculated directly from the geometry to ensure perfect alignment.
- **Level Imputation**: Automatically detects building levels from various column aliases or estimates them from height data.

### Step 2 & 3: Optimization & Clustering
Analyzes the entire dataset (Width, Height, Levels) to find repeating patterns. It groups similar buildings into clusters to reduce the number of unique 3D models required.

### Step 4: Hybrid Assignment
Determines which buildings "fit" a geotypical prototype and which are too unique. 
- Buildings exceeding the `--max_dist` threshold or having low IoU are marked as **Geospecific**.
- This step implements a **Clean Slate Guarantee**, purging old data to ensure no stale artifacts remain.

### Step 5 & 6: 3D Model Generation
- **Geotypical**: Creates one high-quality OBJ model per cluster.
- **Geospecific**: Generates unique OBJ models for every "outlier" building using `trimesh` extrusion, preserving its exact footprint.
- All models are automatically UV-mapped and textured.

### Step 7: Footprint Reconstruction
Generates a `geotypical_footprints.geojson` representing the optimized bounding boxes used for the geotypical buildings, useful for GIS verification.

---

## 📋 Prerequisites

1.  **Python 3.10+**
2.  **Dependencies**:
    ```bash
    pip install geopandas shapely pandas trimesh scikit-learn matplotlib scipy joblib tqdm
    ```
3.  **Blender 4.0+** (for visualization and scene assembly).

---

## 💻 How to Run

### 1. Run the Pipeline
Execute the main script from the project root. You must provide an input file, a model file, and an output directory.

**Standard Run (Clustering):**
```bash
python src/run_pipeline.py export.geojson models/mobb_rf.pkl ./output
```

**GeoPackage with Layer Selection:**
```bash
python src/run_pipeline.py data.gpkg models/mobb_rf.pkl ./output --layer buildings_layer --merge
```
*Note: If a GeoPackage has multiple layers and no `--layer` is specified, the pipeline will list all available layers and default to the first one.*

**High-Precision Run (Merging + MOBB):**
```bash
python src/run_pipeline.py export.geojson models/mobb_rf.pkl ./output --merge --use_mobb --max_dist 10
```

### 2. Import into Blender
The pipeline results are best viewed in Blender using the provided automation script.

```bash
blender -P src/import_city_blender.py
```
*Note: Ensure you update the `CSV_PATH` and `MODELS_DIR` in `import_city_blender.py` before running.*

---

## 📊 Summary & Reporting
At the end of every run, the pipeline generates:
1.  **Console Summary**: A quick table of KPIs (counts, IoU, execution time).
2.  **`summary.md`**: A detailed technical report located in the output directory.

---

## 🛡 Robustness Features
- **File Lock Handling**: The pipeline detects if files are open in GIS/Blender and provides clear instructions.
- **Output Purging**: Every run cleans relevant sub-directories to prevent "ghost buildings" from previous executions.
- **Attribute Preservation**: Spatial joins ensure level and height data are carried through the merging process.
