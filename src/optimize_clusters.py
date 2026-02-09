import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Optimize building clusters (Width/Height).")
    parser.add_argument("input_shp", help="Path to predicted point shapefile")
    parser.add_argument("output_csv", help="Path to save cluster centers CSV")
    parser.add_argument("--n_clusters", type=int, default=100, help="Number of clusters (default: 100)")
    parser.add_argument("--plot", default="clusters.png", help="Path to save plot image")
    parser.add_argument("--min_iou", type=float, default=0.0, help="Minimum IoU (0.0 to 1.0) to include in clustering")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_shp):
        print(f"Error: Input file not found: {args.input_shp}")
        return
        
    print(f"Loading {args.input_shp}...")
    gdf = gpd.read_file(args.input_shp)
    print(f"Loaded {len(gdf)} features.")
    
    # Identify columns (handle truncation)
    width_col = 'pred_width'
    height_col = 'pred_height' if 'pred_height' in gdf.columns else 'pred_heigh'
    iou_col = 'pred_iou'
    levels_col = 'building:levels' # Can be 'building:l' if truncated
    if levels_col not in gdf.columns and 'building_l' in gdf.columns: levels_col = 'building_l'
    
    if width_col not in gdf.columns or height_col not in gdf.columns:
        print(f"Error: Missing columns. Looking for {width_col}, {height_col}")
        return
        
    # Extract data
    cols = [width_col, height_col]
    names = ['width', 'height']
    
    if iou_col in gdf.columns:
        cols.append(iou_col)
        names.append('iou')
        
    if levels_col in gdf.columns:
        cols.append(levels_col)
        names.append('levels')
        
    print(f"Using columns: {cols}")
    data = gdf[cols].copy()
    data.columns = names
    
    if 'iou' not in data.columns:
        print(f"Warning: {iou_col} not found. Proceeding without IoU filtering.")
        data['iou'] = 1.0 # Assume perfect if missing just to pass filter
        
    if 'levels' not in data.columns:
        print(f"Warning: {levels_col} not found. Assuming 1 level for all.")
        data['levels'] = 1.0
        
    # Clean levels (ensure numeric)
    data['levels'] = pd.to_numeric(data['levels'], errors='coerce')
    data['levels'] = data['levels'].fillna(1.0)
        
    # Filter by IoU
    if args.min_iou > 0:
        initial_len = len(data)
        data = data[data['iou'] >= args.min_iou].copy()
        print(f"Filtered by IoU >= {args.min_iou}: {len(data)} / {initial_len} buildings remaining.")
    
    if len(data) < args.n_clusters:
        print(f"Error: Not enough data points ({len(data)}) for {args.n_clusters} clusters.")
        return
    
    # 1. Filter Outliers (Top 1% roughly) to avoid skewing clusters
    # We filter based on Euclidean distance from origin or just max values
    q99_w = data['width'].quantile(0.99)
    q99_h = data['height'].quantile(0.99)
    # Don't filter levels strictly, but maybe crazy high ones? Just cap them maybe?
    # q99_l = data['levels'].quantile(0.99)
    
    mask = (data['width'] <= q99_w) & (data['height'] <= q99_h)
    data_clean = data[mask].copy()
    print(f"Filtered {len(data) - len(data_clean)} outliers (Top 1%). Using {len(data_clean)} features.")
    
    # Use Width, Height, AND Levels for clustering
    X = data_clean[['width', 'height', 'levels']].values
    
    # 2. Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. K-Means
    print(f"Running K-Means with K={args.n_clusters}...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # 4. Analyze Adherence
    # Distance to nearest cluster (in original scale)
    centers_scaled = kmeans.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)
    
    # Assign every point to nearest
    labels = kmeans.predict(X_scaled)
    
    # Calculate error (Euclidean distance in normalized space? Or meaningful physical space?)
    # Usually we want physical meaning, but levels is unitless count vs meters.
    # Scaler effectively weighted them equally by variance. 
    # Let's report raw distance in normalized space for adherence "quality".
    # Or just keep mean error generic.
    
    # assigned center for each point (in original scale)
    assigned_centers = centers[labels]
    
    # For physical error, maybe just use width/height distance?
    dist_wh = np.linalg.norm(X[:, :2] - assigned_centers[:, :2], axis=1)
    
    # 95th Percentile Error (Width/Height only)
    p95_error = np.percentile(dist_wh, 95)
    mean_error = np.mean(dist_wh)
    
    print(f"Optimization Results:")
    print(f"Mean spatial (W/H) Error: {mean_error:.4f} m")
    print(f"95% of buildings are within {p95_error:.4f} m (W/H) of a cluster center.")
    
    # 5. Save Clusters
    df_clusters = pd.DataFrame(centers, columns=['optimum_width', 'optimum_height', 'optimum_levels'])
    # Round levels to nearest integer?
    # Actually, keep float for centroid, but usually levels are int.
    # Let's verify: In assign_clusters, we might need to match exactly? No, usually we assign to nearest cluster.
    # But for creating OBJ, we probably want integer levels.
    df_clusters['optimum_levels'] = df_clusters['optimum_levels'].round().astype(int)
    # Ensure at least 1
    df_clusters['optimum_levels'] = df_clusters['optimum_levels'].clip(lower=1)
    
    df_clusters = df_clusters.sort_values(by='optimum_width')
    df_clusters.to_csv(args.output_csv, index=False)
    print(f"Cluster centers saved to {args.output_csv}")
    
    # 6. Plot
    print(f"Generating plot to {args.plot}...")
    plt.figure(figsize=(10, 8))
    
    # Plot a sample of background points to avoid overcrowding
    sample_indices = np.random.choice(len(X), size=min(10000, len(X)), replace=False)
    plt.scatter(X[sample_indices, 0], X[sample_indices, 1], c='lightgray', s=5, alpha=0.5, label='Buildings (Sample)')
    
    # Plot Clusters
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=50, marker='x', label='Optimum Clusters')
    
    plt.title(f"Optimum {args.n_clusters} Width/Height Clusters\n95% Adherence Radius: {p95_error:.2f}m")
    plt.xlabel("Width (m)")
    plt.ylabel("Height (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(args.plot)
    print("Done.")

if __name__ == "__main__":
    main()
