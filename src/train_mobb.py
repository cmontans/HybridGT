import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
import math
from tqdm import tqdm

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


def extract_features(gdf):
    """
    Extracts geometric features from the GeoDataFrame.
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
    
    # Orientation Feature: Angle of the longest edge
    print("Calculating longest edge angles...")
    tqdm.pandas(desc="Features")
    features['longest_edge_angle'] = gdf.geometry.progress_apply(get_longest_edge_angle)
    
    # Transform input angle feature to sin/cos
    features['sin_2alpha_in'] = np.sin(2 * np.radians(features['longest_edge_angle']))
    features['cos_2alpha_in'] = np.cos(2 * np.radians(features['longest_edge_angle']))
    
    return features

def train_model(gdf, model_path):
    """Train the MOBB Random Forest model from a GeoDataFrame of building footprints.

    The model is self-supervised: MOBB parameters are computed from the input
    geometries and used as ground truth targets.

    Args:
        gdf: GeoDataFrame with building polygon geometries.
        model_path: Destination path for the saved model (.pkl).

    Returns:
        model_path on success, None if there are not enough valid polygons.
    """
    print(f"Auto-training model from {len(gdf)} input features...")

    # Preprocessing
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf['geometry'] = gdf.geometry.apply(make_valid)
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geometry.type == 'Polygon'].copy()

    print(f"After preprocessing: {len(gdf)} polygons.")

    if len(gdf) < 10:
        print(f"Error: Not enough valid building polygons to train a model "
              f"(found {len(gdf)}, need at least 10).")
        return None

    print("Reprojecting to EPSG:32630 (UTM Zone 30N)...")
    if gdf.crs is None:
        print("Warning: Source CRS is missing. Assuming EPSG:4326.")
        gdf = gdf.set_crs(epsg=4326)
    gdf = gdf.to_crs(epsg=32630)

    if len(gdf) > 20000:
        print("Dataset large, sampling 20000 polygons for training...")
        gdf = gdf.sample(20000, random_state=42)

    # Generate Ground Truth (MOBB)
    print("Generating ground truth (MOBB)...")
    mobb_results = gdf.geometry.apply(get_mobb_params)
    y_raw = pd.DataFrame(mobb_results.tolist(), columns=['width', 'height', 'angle'], index=gdf.index)

    mask = (y_raw['width'] > 0) & (y_raw['height'] > 0)
    gdf = gdf[mask]
    y_raw = y_raw[mask]

    # Transform targets: encode angle as sin/cos
    y = pd.DataFrame(index=y_raw.index)
    y['width'] = y_raw['width']
    y['height'] = y_raw['height']
    rads = np.radians(y_raw['angle'])
    y['sin_2a'] = np.sin(2 * rads)
    y['cos_2a'] = np.cos(2 * rads)

    print(f"Training on {len(gdf)} samples...")

    # Feature Engineering
    X = extract_features(gdf)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Quick evaluation
    y_pred_raw = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred_raw, index=y_test.index, columns=y_test.columns)
    mae_w = mean_absolute_error(y_test['width'], y_pred_df['width'])
    mae_h = mean_absolute_error(y_test['height'], y_pred_df['height'])
    y_test_angle = np.degrees(0.5 * np.arctan2(y_test['sin_2a'], y_test['cos_2a']))
    y_pred_angle = np.degrees(0.5 * np.arctan2(y_pred_df['sin_2a'], y_pred_df['cos_2a']))
    angle_diff = np.abs(y_test_angle - y_pred_angle)
    angle_diff = np.minimum(angle_diff, 180 - angle_diff)
    mae_a = np.mean(angle_diff)
    print(f"Training complete — MAE Width: {mae_w:.2f} m | MAE Height: {mae_h:.2f} m | MAE Angle: {mae_a:.2f} deg")

    # Save model
    model_dir = os.path.dirname(os.path.abspath(model_path))
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model_path


def main():
    # Paths
    print("Initializing...")
    shapefile_path = os.path.join("cantabria-260206-free.shp", "gis_osm_buildings_a_free_1.shp")
    model_path = os.path.join("models", "mobb_rf.pkl")

    # 1. Load Data
    print(f"Loading shapefile from {shapefile_path}...")
    if not os.path.exists(shapefile_path):
        print(f"Error: Shapefile not found at {shapefile_path}")
        return

    gdf = gpd.read_file(shapefile_path)
    print(f"Loaded {len(gdf)} polygons.")

    result = train_model(gdf, model_path)
    if result is None:
        return

    # Reload for visualization (model already saved inside train_model)
    gdf_proj = gdf[gdf.geometry.type == 'Polygon'].copy()
    gdf_proj['geometry'] = gdf_proj.geometry.apply(make_valid)
    gdf_proj = gdf_proj[~gdf_proj.geometry.is_empty]
    if gdf_proj.crs is None:
        gdf_proj = gdf_proj.set_crs(epsg=4326)
    gdf_proj = gdf_proj.to_crs(epsg=32630)
    if len(gdf_proj) > 20000:
        gdf_proj = gdf_proj.sample(20000, random_state=42)

    mobb_results = gdf_proj.geometry.apply(get_mobb_params)
    y_raw = pd.DataFrame(mobb_results.tolist(), columns=['width', 'height', 'angle'], index=gdf_proj.index)
    mask = (y_raw['width'] > 0) & (y_raw['height'] > 0)
    gdf_proj = gdf_proj[mask]
    y_raw = y_raw[mask]

    y = pd.DataFrame(index=y_raw.index)
    y['width'] = y_raw['width']
    y['height'] = y_raw['height']
    rads = np.radians(y_raw['angle'])
    y['sin_2a'] = np.sin(2 * rads)
    y['cos_2a'] = np.cos(2 * rads)

    X = extract_features(gdf_proj)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load(model_path)
    y_pred_raw = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred_raw, index=y_test.index, columns=y_test.columns)
    y_pred_angle = np.degrees(0.5 * np.arctan2(y_pred_df['sin_2a'], y_pred_df['cos_2a']))
    y_test_angle = np.degrees(0.5 * np.arctan2(y_test['sin_2a'], y_test['cos_2a']))

    y_pred_final = y_pred_df.copy()
    y_pred_final['angle'] = y_pred_angle
    y_test_final = y_test.copy()
    y_test_final['angle'] = y_test_angle

    print("Generating visualization...")
    plot_results(gdf_proj.loc[y_test.index], y_test_final, y_pred_final)

def plot_results(gdf_test, y_test, y_pred):
    n_samples = 4
    indices = np.random.choice(len(gdf_test), n_samples, replace=False)
    
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 4))
    
    for i, idx in enumerate(indices):
        real_idx = gdf_test.index[idx]
        geom = gdf_test.geometry.loc[real_idx]
        
        # True values
        w_true = y_test.iloc[idx]['width']
        h_true = y_test.iloc[idx]['height']
        a_true = y_test.iloc[idx]['angle']
        
        # Pred values
        w_pred = y_pred.iloc[idx]['width']
        h_pred = y_pred.iloc[idx]['height']
        a_pred = y_pred.iloc[idx]['angle']
        
        ax = axes[i]
        
        x, y = geom.exterior.xy
        ax.plot(x, y, 'b-', linewidth=2, label='Original')
        
        centroid = geom.centroid
        rect_pred = create_rect(centroid.x, centroid.y, w_pred, h_pred, a_pred)
        xr, yr = rect_pred.exterior.xy
        ax.plot(xr, yr, 'r--', linewidth=2, label='Pred')
        
        # True Box (for comparison)
        rect_true = create_rect(centroid.x, centroid.y, w_true, h_true, a_true)
        xt, yt = rect_true.exterior.xy
        ax.plot(xt, yt, 'g:', linewidth=1, label='True')
        
        ax.set_title(f"A_True: {a_true:.0f}°\nA_Pred: {a_pred:.0f}°")
        ax.axis('equal')
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig("mobb_prediction_improved.png")
    print("Visualization saved to mobb_prediction_improved.png")

def create_rect(cx, cy, w, h, angle_deg):
    from shapely.affinity import rotate, translate
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

if __name__ == "__main__":
    main()
