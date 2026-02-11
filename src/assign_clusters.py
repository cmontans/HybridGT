import geopandas as gpd
import pandas as pd
import numpy as np
import argparse
import os
from scipy.spatial import cKDTree

def main():
    parser = argparse.ArgumentParser(description="Assign buildings to nearest cluster models.")
    parser.add_argument("predictions_file", help="Path to predictions file (Shapefile, GeoJSON, or GeoPackage)")
    parser.add_argument("clusters_csv", help="Path to clusters CSV")
    parser.add_argument("output_csv", help="Path to output CSV for Blender import")
    parser.add_argument("--geospecific_geojson", help="Path to save geospecific building footprints", default=None)
    parser.add_argument("--max_dist", type=float, default=9999.0, help="Maximum fit distance to allow clustering. If exceeded, building is geospecific.")
    
    args = parser.parse_args()
    
    # 1. Load Clusters
    print(f"Loading clusters from {args.clusters_csv}...")
    df_clusters = pd.read_csv(args.clusters_csv)
    
    # Check if levels are present
    has_levels = 'optimum_levels' in df_clusters.columns
    
    # Create KDTree for fast nearest neighbor search
    if has_levels:
        cluster_points = df_clusters[['optimum_width', 'optimum_height', 'optimum_levels']].values
    else:
        cluster_points = df_clusters[['optimum_width', 'optimum_height']].values
        
    cluster_dims_count = cluster_points.shape[1]
    cluster_tree = cKDTree(cluster_points)
    
    # Generate OBJ filenames for clusters
    cluster_obj_names = []
    for i, row in df_clusters.iterrows():
        w = row['optimum_width']
        h = row['optimum_height']
        
        if has_levels:
            l = int(row['optimum_levels'])
            fname = f"cluster_{i}_w{w:.2f}_h{h:.2f}_l{l}.obj"
        else:
            fname = f"cluster_{i}_w{w:.2f}_h{h:.2f}.obj"
            
        cluster_obj_names.append(fname)
        
    df_clusters['obj_filename'] = cluster_obj_names
    
    # 2. Load Predictions
    print(f"Loading predictions from {args.predictions_file}...")
    gdf_buildings = gpd.read_file(args.predictions_file)
    
    print(f"Loaded {len(gdf_buildings)} buildings.")
    
    if len(gdf_buildings) == 0:
        print("No buildings found in the input file.")
        return
        
    # Check for levels in input
    levels_col = 'building:levels'
    if levels_col not in gdf_buildings.columns and 'building_l' in gdf_buildings.columns:
        levels_col = 'building_l'
        
    if has_levels and levels_col not in gdf_buildings.columns:
        print(f"Warning: Clusters have levels but input file missing '{levels_col}'. Assuming 1.")
        gdf_buildings[levels_col] = 1

    # 4. Assign Clusters
    print("Assigning nearest clusters...")
    
    # Extract width/height from predictions
    w_col = 'pred_width'
    h_col = 'pred_height'
    a_col = 'pred_angle'
    
    if h_col not in gdf_buildings.columns and 'pred_heigh' in gdf_buildings.columns:
        h_col = 'pred_heigh'
        
    required_cols = [w_col, h_col, a_col]
    for col in required_cols:
        if col not in gdf_buildings.columns:
            print(f"Error: Missing column '{col}' in file. Available: {gdf_buildings.columns.tolist()}")
            return
            
    # Prepare query points
    if has_levels:
        # Ensure numeric
        gdf_buildings[levels_col] = pd.to_numeric(gdf_buildings[levels_col], errors='coerce').fillna(1)
        query_points = gdf_buildings[[w_col, h_col, levels_col]].values
    else:
        query_points = gdf_buildings[[w_col, h_col]].values
    
    # Vectorized "Closest Smaller" Logic
    print("Calculating distances to all clusters...")
    cluster_dims = cluster_points # (M, D)
    building_dims = query_points  # (N, D)
    
    # Broadcast to calculate squared euclidean distance
    # (N, 1, D) - (1, M, D)
    diff = building_dims[:, np.newaxis, :] - cluster_dims[np.newaxis, :, :]
    
    # Weighted distance? Levels (1-10) vs Width (10-50).
    # Width difference of 1m is significant. Level difference of 1 is significant.
    # 1 level ~= 3m. So difference of 1 level is like 3m difference.
    # If we treat levels as raw, a difference of 1 is smaller than difference of 3m.
    # So levels will have less impact than width/height if unscaled.
    # But usually we want the levels to match closely.
    # Let's multiply levels difference by 3 (approx height per floor) to weight it?
    # Actually 'optimize_clusters' used StandardScaler so it found clusters considering variance.
    # Here we assign based on raw Euclid.
    # If we don't scale, 'closest' might pick a cluster with right width but wrong levels over one with wrong width but right levels.
    # Let's keep it simple (raw Euclidean) for now, as consistent with previous logic.
    
    dist_sq = np.sum(diff**2, axis=2) # (N, M)
    
    # Check "Smaller or Equal" condition
    cond_w = cluster_dims[np.newaxis, :, 0] <= building_dims[:, np.newaxis, 0]
    cond_h = cluster_dims[np.newaxis, :, 1] <= building_dims[:, np.newaxis, 1]
    
    if has_levels:
        cond_l = cluster_dims[np.newaxis, :, 2] <= building_dims[:, np.newaxis, 2]
        is_smaller = cond_w & cond_h & cond_l
    else:
        is_smaller = cond_w & cond_h
    
    # Mask distances where cluster is NOT smaller
    masked_dist = np.where(is_smaller, dist_sq, np.inf)
    
    # Find best index (closest among smaller)
    best_idx = np.argmin(masked_dist, axis=1)
    min_vals = np.min(masked_dist, axis=1)
    
    # Handle Fallback
    fallback_mask = np.isinf(min_vals)
    fallback_count = np.sum(fallback_mask)
    
    # Identify Geospecific
    distances = np.sqrt(min_vals)
    
    if fallback_count > 0:
        # Fallback: building size is smaller than all clusters.
        # Force fit to closest
        fallback_indices = np.argmin(dist_sq[fallback_mask], axis=1)
        best_idx[fallback_mask] = fallback_indices
        distances[fallback_mask] = np.sqrt(np.min(dist_sq[fallback_mask], axis=1))
        
    indices = best_idx
    
    # Determine Clustered vs Geospecific
    is_geospecific = np.zeros(len(gdf_buildings), dtype=bool)
    
    if args.geospecific_geojson:
        # 1. Distance > max_dist
        mask_dist = distances > args.max_dist
        is_geospecific = mask_dist
        print(f"Identified {np.sum(is_geospecific)} geospecific buildings (Dist > {args.max_dist}).")
        
    # 5. Prepare Output
    output_data = []
    
    # Indices of clustered
    clustered_indices = np.where(~is_geospecific)[0]
    
    for i in clustered_indices:
        idx = indices[i]
        dist = distances[i]
        
        # Original building data
        building = gdf_buildings.iloc[i]
        
        # Cluster data
        cluster_file = df_clusters.iloc[idx]['obj_filename']
        cluster_levels = int(df_clusters.iloc[idx]['optimum_levels']) if has_levels else 1
        
        # Position (use centroid of polygon for placement)
        geom = building.geometry
        centroid = geom.centroid
        x = centroid.x
        y = centroid.y
        z = 0 # Ground
        
        try:
             angle_deg = building[a_col]
        except:
             angle_deg = 0
        
        entry = {
            'x': x,
            'y': y,
            'z': z,
            'angle_deg': angle_deg,
            'obj_filename': cluster_file,
            'pred_width': building[w_col],
            'pred_height': building[h_col],
            'fit_dist': dist
        }
        if has_levels:
            entry['pred_levels'] = building[levels_col]
            entry['cluster_levels'] = cluster_levels
            
        output_data.append(entry)
        
    df_out = pd.DataFrame(output_data)
    
    print(f"Saving {len(df_out)} clustered instances to {args.output_csv}...")
    df_out.to_csv(args.output_csv, index=False)
    
    # Save Geospecific
    if args.geospecific_geojson:
        geospecific_gdf = gdf_buildings[is_geospecific].copy()
        try:
            # On Windows, sometimes file handles are held. Try deleting if exists.
            if os.path.exists(args.geospecific_geojson):
                try:
                    os.remove(args.geospecific_geojson)
                except PermissionError:
                    print(f"Warning: Could not remove {args.geospecific_geojson}. It might be locked.")

            if len(geospecific_gdf) > 0:
                print(f"Saving {len(geospecific_gdf)} geospecific footprints to {args.geospecific_geojson}...")
                # Try saving. Using 'fiona' engine can sometimes be more robust to locks than 'pyogrio' on Windows.
                try:
                    geospecific_gdf.to_file(args.geospecific_geojson, driver='GeoJSON')
                except Exception as e:
                    print(f"Error saving with default engine: {e}. Trying 'fiona'...")
                    geospecific_gdf.to_file(args.geospecific_geojson, driver='GeoJSON', engine='fiona')
            else:
                print("No geospecific buildings found.")
                # Create empty file
                with open(args.geospecific_geojson, 'w') as f:
                    f.write('{"type": "FeatureCollection", "features": []}')
        except PermissionError:
            print("\n" + "!"*60)
            print(f"CRITICAL ERROR: Permission denied when writing to {args.geospecific_geojson}")
            print("Please ensure the file is NOT open in QGIS, Blender, or any other program.")
            print("!"*60 + "\n")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error saving geospecific footprints: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
