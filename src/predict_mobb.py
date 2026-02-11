import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
import os
import math
import argparse
import pyogrio
from tqdm import tqdm
from shapely.geometry import Point, Polygon
from shapely.validation import make_valid

def get_longest_edge_angle(geom):
    """
    Calculates the angle of the longest edge of the polygon.
    Returns angle in degrees [-90, 90].
    """
    if geom is None or geom.is_empty:
        return 0
        
    try:
        coords = list(geom.exterior.coords)
    except AttributeError:
        return 0
        
    if len(coords) < 2:
        return 0
        
    max_len = -1
    best_angle = 0
    
    for i in range(len(coords) - 1):
        p0 = np.array(coords[i])
        p1 = np.array(coords[i+1])
        d = np.linalg.norm(p1 - p0)
        
        if d > max_len:
            max_len = d
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            best_angle = math.degrees(math.atan2(dy, dx))
            
    # Normalize to [-90, 90]
    if best_angle < -90:
        best_angle += 180
    elif best_angle >= 90:
        best_angle -= 180
        
    return best_angle


def get_mobb_params(geom):
    """
    Calculates the Minimum Oriented Bounding Box (MOBB) parameters
    (width, height, angle) for a given geometry.
    Returns: width, height, angle (in degrees)
    """
    if geom is None or geom.is_empty:
        return 0, 0, 0
    
    mobb = geom.minimum_rotated_rectangle
    try:
        coords = list(mobb.exterior.coords)
    except AttributeError:
        return 0, 0, 0

    if len(coords) < 4:
        return 0, 0, 0

    p0 = np.array(coords[0])
    p1 = np.array(coords[1])
    p2 = np.array(coords[2])
    
    d1 = np.linalg.norm(p1 - p0)
    d2 = np.linalg.norm(p2 - p1)
    
    if d1 >= d2:
        width = d1
        height = d2
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
    else:
        width = d2
        height = d1
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
    
    angle = math.degrees(math.atan2(dy, dx))
    
    # Normalize angle to [-90, 90]
    if angle < -90:
        angle += 180
    elif angle >= 90:
        angle -= 180
        
    return width, height, angle


def extract_features(gdf):
    """
    Extracts geometric features from the GeoDataFrame.
    Must match the training script exactly.
    """
    features = pd.DataFrame()
    
    # Basic geometry features
    features['area'] = gdf.geometry.area
    features['perimeter'] = gdf.geometry.length
    
    # Convex hull features
    hull = gdf.geometry.convex_hull
    features['hull_area'] = hull.area
    features['hull_perimeter'] = hull.length
    
    # Shape descriptors
    features['compactness'] = (4 * np.pi * features['area']) / (features['perimeter']**2 + 1e-6)
    features['elongation'] = features['hull_perimeter'] / (features['hull_area'].apply(np.sqrt) + 1e-6)
    
    # Orientation Feature
    print("Calculating longest edge angles...")
    tqdm.pandas(desc="Features")
    features['longest_edge_angle'] = gdf.geometry.progress_apply(get_longest_edge_angle)
    
    # Transform input angle feature
    features['sin_2alpha_in'] = np.sin(2 * np.radians(features['longest_edge_angle']))
    features['cos_2alpha_in'] = np.cos(2 * np.radians(features['longest_edge_angle']))
    
    return features

def main():
    parser = argparse.ArgumentParser(description="Predict MOBB parameters from building polygons (SHP/GeoJSON).")
    parser.add_argument("input_file", help="Path to input polygon file (shp, json, geojson, gpkg)")
    parser.add_argument("output_file", help="Path to output point file (shp, json, geojson)")
    parser.add_argument("--model", default=os.path.join("models", "mobb_rf.pkl"), help="Path to trained model")
    parser.add_argument("--use_mobb", action="store_true", help="Calculate dimensions using MOBB instead of predicting")
    parser.add_argument("--layer", help="Layer name to read from multi-layer files (e.g. GPKG)")
    
    args = parser.parse_args()
    
    # Load Model
    model = None
    if not args.use_mobb:
        if not os.path.exists(args.model):
            print(f"Error: Model file not found at {args.model}")
            return
        print(f"Loading model from {args.model}...")
        try:
            model = joblib.load(args.model)
        except Exception as e:
            print(f"Error loading model: {e}")
            return

    print(f"Loading data from {args.input_file}...")
    
    # Check for multi-layer GPKG
    layer = args.layer
    if args.input_file.lower().endswith(".gpkg"):
        try:
            layers = pyogrio.list_layers(args.input_file)
            layer_names = layers[:, 0] if len(layers) > 0 else []
            
            if len(layer_names) > 1 and not layer:
                print("\n" + "!"*60)
                print(f"WARNING: Multiple layers detected in {os.path.basename(args.input_file)}:")
                for ln in layer_names:
                    print(f"  - {ln}")
                print(f"No --layer specified. Using the first layer: '{layer_names[0]}'")
                print("!"*60 + "\n")
                layer = layer_names[0]
            elif layer and layer not in layer_names:
                print(f"Error: Layer '{layer}' not found in {args.input_file}. Available: {list(layer_names)}")
                return
        except Exception as e:
            print(f"Warning: Could not list layers in {args.input_file}: {e}")

    try:
        gdf = gpd.read_file(args.input_file, layer=layer)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
        
    print(f"Loaded {len(gdf)} features.")
    
    # Preprocessing
    print(f"Original geometry types: {gdf.geometry.type.value_counts()}")
    
    # Explode MultiPolygons
    gdf = gdf.explode(index_parts=True)
    print(f"After explode: {len(gdf)} features.")
    
    # Ensure Polygons
    gdf = gdf[gdf.geometry.type == 'Polygon'].copy()
    print(f"After Polygon filter: {len(gdf)} features.")
    
    gdf['geometry'] = gdf.geometry.apply(make_valid)
    gdf = gdf[~gdf.geometry.is_empty]
    print(f"After validation: {len(gdf)} features.")
    
    if len(gdf) == 0:
        print("Error: No valid polygons found after preprocessing.")
        return
    
    # Reprojection (CRITICAL)
    target_crs = "EPSG:32630" # UTM Zone 30N
    print(f"Reprojecting to {target_crs}...")
    
    if gdf.crs is None:
        print("Warning: Source CRS is missing. Assuming EPSG:4326.")
        gdf.set_crs(epsg=4326, inplace=True)
        
    gdf_projected = gdf.to_crs(target_crs)
    
    # Feature Extraction
    print("Extracting features...")
    X = extract_features(gdf_projected)
    
    # Calculate MOBB Parameters for EVERY building (as requested for orientation)
    print("Calculating MOBB parameters...")
    mobb_results = gdf_projected.geometry.apply(get_mobb_params)
    df_mobb = pd.DataFrame(mobb_results.tolist(), columns=['mobb_width', 'mobb_height', 'mobb_angle'])
    
    # Predict or Calculate MOBB
    if args.use_mobb:
        print("Using MOBB dimensions...")
        df_preds = df_mobb.rename(columns={'mobb_width': 'pred_width', 'mobb_height': 'pred_height', 'mobb_angle': 'pred_angle'})
        # Add sin/cos for safety
        df_preds['sin_2a'] = np.sin(2 * np.radians(df_preds['pred_angle']))
        df_preds['cos_2a'] = np.cos(2 * np.radians(df_preds['pred_angle']))
    else:
        print("Predicting MOBB dimensions (but using calculated orientation)...")
        # Expected output: width, height, sin_2a, cos_2a
        preds = model.predict(X)
        df_preds = pd.DataFrame(preds, columns=['pred_width', 'pred_height', 'sin_2a', 'cos_2a'])
        # Override predicted angle with ACTUAL calculated angle
        df_preds['pred_angle'] = df_mobb['mobb_angle']

    # --- Impute Missing Attributes (Building Type & Levels) ---
    print("Imputing missing 'building' and 'building:levels'...")
    
    # 1. Building Type
    if 'building' not in gdf.columns:
        gdf['building'] = None
        
    # Ensure column is object type to allow updates
    gdf['building'] = gdf['building'].astype('object')
        
    # Filter for valid training data (exclude 'yes' to learn specific types if possible)
    # We treat 'yes' as generic, but if we don't have enough specific types, we might need to include it.
    # Let's try to train on specific types first.
    valid_types = gdf[gdf['building'].notna() & (gdf['building'] != 'yes') & (gdf['building'] != '')]
    if len(valid_types) > 50:
        print(f"Training building type classifier on {len(valid_types)} specific samples...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        le = LabelEncoder()
        # Convert to string to ensure consistency
        y_type_train = le.fit_transform(valid_types['building'].astype(str))
        X_type_train = X.loc[valid_types.index]
        
        clf_type = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_type.fit(X_type_train, y_type_train)
        
        # Predict for missing values (NaN or empty)
        missing_type_mask = gdf['building'].isna() | (gdf['building'] == '')
        if missing_type_mask.any():
            X_type_missing = X.loc[missing_type_mask]
            pred_types = clf_type.predict(X_type_missing)
            
            # Inverse transform returns original labels (strings).
            # Convert to list or numpy array of objects to avoid pandas complaints if dtype is picky
            pred_labels = le.inverse_transform(pred_types)
            gdf.loc[missing_type_mask, 'building'] = pred_labels
            print(f"Imputed {missing_type_mask.sum()} missing building types.")
    else:
        print("Not enough specific building type data to train classifier. Skipping type imputation.")
        # If completely missing, default to 'yes'
        if gdf['building'].isna().all():
             gdf['building'] = 'yes'

    # 2. Building Levels Detection (Enhanced)
    levels_col = 'building:levels'
    possible_levels_cols = ['building:levels', 'building_levels', 'building_l', 'levels', 'L']
    detected_col = None
    for c in possible_levels_cols:
        if c in gdf.columns:
            detected_col = c
            break
            
    if detected_col and detected_col != levels_col:
        print(f"Found levels in column '{detected_col}'. Mapping to '{levels_col}'.")
        gdf[levels_col] = gdf[detected_col]
        
    if levels_col not in gdf.columns:
        # Check for height as fallback
        height_col = None
        for c in ['height', 'ele', 'altitude']:
            if c in gdf.columns:
                height_col = c
                break
        
        if height_col:
            print(f"Found height in '{height_col}'. Estimating levels...")
            try:
                h_vals = pd.to_numeric(gdf[height_col], errors='coerce').fillna(3.5) # Default 3.5m if nan
                gdf[levels_col] = np.maximum(1, np.round(h_vals / 3.5)).astype(int)
            except:
                gdf[levels_col] = None
        else:
            gdf[levels_col] = None
            
    # Ensure column is object type
    gdf[levels_col] = gdf[levels_col].astype('object')
        
    # Helper to clean levels
    def clean_levels(val):
        if pd.isna(val) or val == '':
            return np.nan
        try:
            # Handle float directly
            if isinstance(val, (int, float)):
                return float(val)
            # Handle strings like "3", "3.5"
            val_str = str(val).replace(',', '.')
            # Handle ranges "3-5" -> mean? max? let's take mean.
            if '-' in val_str:
                parts = val_str.split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            return float(val_str)
        except:
            return np.nan

    gdf['levels_clean'] = gdf['building:levels'].apply(clean_levels)
    
    valid_levels = gdf[gdf['levels_clean'].notna()]
    if len(valid_levels) > 20:
        print(f"Training levels regressor on {len(valid_levels)} samples...")
        from sklearn.ensemble import RandomForestRegressor
        
        y_levels_train = valid_levels['levels_clean']
        X_levels_train = X.loc[valid_levels.index]
        
        reg_levels = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        reg_levels.fit(X_levels_train, y_levels_train)
        
        # Predict for missing
        missing_levels_mask = gdf['levels_clean'].isna()
        if missing_levels_mask.any():
            X_levels_missing = X.loc[missing_levels_mask]
            pred_levels = reg_levels.predict(X_levels_missing)
            # Round to nearest int, ensure at least 1
            pred_levels = np.maximum(1, np.round(pred_levels)).astype(int)
            # Convert to string or int, assign back to original column
            # If original column has strings mixed with ints, better safe than sorry: cast to int then string?
            # Or just int? Since we cast to object, int is fine.
            gdf.loc[missing_levels_mask, 'building:levels'] = pred_levels
            print(f"Imputed {missing_levels_mask.sum()} missing building levels.")
    else:
        print("Not enough levels data to train regressor. Defaulting to 1.")
        gdf.loc[gdf['building:levels'].isna(), 'building:levels'] = 1
        
    # Drop temporary column
    if 'levels_clean' in gdf.columns:
        gdf.drop(columns=['levels_clean'], inplace=True)

    # ---------------------------------------------------------
    
    # Process predictions (Width, Height, Angle)
    # Already processed above
    
    # Create Output GeoDataFrame
    # We want Point features (Centroids)
    # Use the projected centroids so they are in meters, then maybe project back?
    # Usually output should match input CRS or requested CRS. 
    # Let's keep output in the projected CRS (UTM) as the units (width/height) are in meters.
    # OR project back to original WGS84 but keep attributes in meters.
    # Let's output in UTM to be consistent with the metric attributes.
    
    from shapely.geometry import Polygon
    from shapely.affinity import rotate, translate

    def create_rect(cx, cy, w, h, angle_deg):
        box = Polygon([
            (-w/2, -h/2),
            (w/2, -h/2),
            (w/2, h/2),
            (-w/2, h/2),
            (-w/2, -h/2)
        ])
        box = rotate(box, angle_deg, origin=(0,0))
        box = translate(box, cx, cy)
        return box

    # Calculate IoU
    print("Calculating Fitting Percentage (IoU)...")
    ious = []
    
    # Iterate through predictions
    # gdf_projected matches df_preds in index and length
    for idx, row in tqdm(df_preds.iterrows(), total=df_preds.shape[0], desc="Calculating IoU"):
        # Original geometry
        # Reset index of gdf_projected to ensure alignment if indices differ
        # But gdf_projected comes from gdf which might have gaps?
        # Actually in main, we did: gdf = gdf[...].copy(). 
        # extract_features returns X with same index.
        # model.predict returns array. df_preds created with default index 0..N.
        # So we need to access gdf_projected by integer position or reset index.
        
        geom = gdf_projected.geometry.iloc[idx]
        
        w_pred = row['pred_width']
        h_pred = row['pred_height']
        a_pred = row['pred_angle']
        
        # Centroid
        centroid = geom.centroid
        
        # Reconstruct Prediction
        pred_box = create_rect(centroid.x, centroid.y, w_pred, h_pred, a_pred)
        
        # Validate
        if not geom.is_valid: geom = make_valid(geom)
        if not pred_box.is_valid: pred_box = make_valid(pred_box)
        
        try:
            intersection = geom.intersection(pred_box).area
            union = geom.union(pred_box).area
            iou = intersection / union if union > 0 else 0
        except:
            iou = 0
            
        ious.append(iou)
        
    df_preds['pred_iou'] = ious
    print(f"Average IoU: {np.mean(ious):.4f}")

    print("Creating output shapefile...")
    out_gdf = gpd.GeoDataFrame(
        df_preds[['pred_width', 'pred_height', 'pred_angle', 'pred_iou']], 
        geometry=gdf_projected.geometry.values, # Keep original polygon geometry for geospecific extrusion
        crs=gdf_projected.crs
    )
    
    # Add imputed columns
    out_gdf['building'] = gdf['building'].values
    out_gdf['building:levels'] = gdf['building:levels'].values
    
    # Save
    print(f"Saving to {args.output_file}...")
    
    # Determine driver based on extension if needed, but gpd handles it well
    # For .json/.geojson, driver is "GeoJSON"
    out_ext = os.path.splitext(args.output_file)[1].lower()
    driver = None
    if out_ext in ['.json', '.geojson']:
        driver = "GeoJSON"
        
    out_gdf.to_file(args.output_file, driver=driver)
    print("Done.")

if __name__ == "__main__":
    main()
