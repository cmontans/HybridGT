import geopandas as gpd
import matplotlib.pyplot as plt
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Plot histograms of predicted MOBB attributes.")
    parser.add_argument("input_shp", help="Path to predicted point shapefile")
    parser.add_argument("output_img", help="Path to save histogram image (e.g., histograms.png)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_shp):
        print(f"Error: Input file not found: {args.input_shp}")
        return
        
    print(f"Loading {args.input_shp}...")
    try:
        gdf = gpd.read_file(args.input_shp)
    except Exception as e:
        print(f"Error reading shapefile: {e}")
        return

    print(f"Loaded {len(gdf)} features.")
    print("Columns:", gdf.columns.tolist())
    
    # Identify columns
    # potential names due to shapefile truncation
    width_col = 'pred_width'
    height_col = 'pred_height' if 'pred_height' in gdf.columns else 'pred_heigh'
    angle_col = 'pred_angle'
    
    missing_cols = []
    for col in [width_col, height_col, angle_col]:
        if col not in gdf.columns:
            missing_cols.append(col)
            
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print("Available columns:", gdf.columns.tolist())
        return

    # Plot
    print("Generating histograms...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    # Width
    axes[0].hist(gdf[width_col], bins=50, color='skyblue', edgecolor='black')
    axes[0].set_title('Predicted Width Distribution')
    axes[0].set_xlabel('Width (m)')
    axes[0].set_ylabel('Frequency')
    
    # Height
    axes[1].hist(gdf[height_col], bins=50, color='lightgreen', edgecolor='black')
    axes[1].set_title('Predicted Height Distribution')
    axes[1].set_xlabel('Height (m)')
    axes[1].set_ylabel('Frequency')
    
    # Angle
    axes[2].hist(gdf[angle_col], bins=50, color='salmon', edgecolor='black')
    axes[2].set_title('Predicted Angle Distribution')
    axes[2].set_xlabel('Angle (degrees)')
    axes[2].set_ylabel('Frequency')
    
    # IoU / Fitting Percentage
    iou_col = 'pred_iou'
    if iou_col in gdf.columns:
        # Plot percentage
        axes[3].hist(gdf[iou_col] * 100, bins=50, color='mediumpurple', edgecolor='black')
        axes[3].set_title('Fitting Percentage Distribution')
        axes[3].set_xlabel('Fitting % (IoU * 100)')
        axes[3].set_ylabel('Frequency')
    else:
        axes[3].axis('off')
        print(f"Warning: {iou_col} not found. Skipping IoU histogram.")
    
    plt.tight_layout()
    plt.savefig(args.output_img)
    print(f"Histograms saved to {args.output_img}")

if __name__ == "__main__":
    main()
