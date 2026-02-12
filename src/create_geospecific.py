import geopandas as gpd
import pandas as pd
import numpy as np
import trimesh
import argparse
import os
import shapely.geometry
from shapely.ops import triangulate

def write_textured_obj(filename, polygon, height, mtl_filename=None):
    """
    Manually writes an OBJ for an extruded footprint with facade and roof textures.
    - Sides (Facades): U = cumulative perimeter / 10.0, V = actual height / 10.0.
    - Top (Roof): U = X / 10.0, V = Y / 10.0 (10m x 10m scale).
    """
    if polygon.is_empty:
        return False

    # Ensure it's a Polygon
    if isinstance(polygon, shapely.geometry.MultiPolygon):
        polygon = max(polygon.geoms, key=lambda a: a.area)
    
    # Exterior ring coords
    coords = list(polygon.exterior.coords)
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    
    n_base = len(coords)
    
    # Roof triangulation for the top face
    try:
        roof_v2, roof_faces = trimesh.creation.triangulate_polygon(polygon)
    except Exception as e:
        print(f"  Triangulation failed: {e}. Falling back to untextured export.")
        return False

    with open(filename, 'w') as f:
        f.write("# Textured Geospecific Building\n")
        if mtl_filename:
            f.write(f"mtllib {mtl_filename}\n")
        f.write("o Building\n")

        # --- Vertices & UVs for Sides ---
        
        # 1. Write Side Vertices
        v_offset = 0
        for x, y in coords:
            f.write(f"v {x:.4f} {y:.4f} 0.0000\n")         # Bottom
            f.write(f"v {x:.4f} {y:.4f} {height:.4f}\n")   # Top
        
        # 2. Write Side UVs (10m x 10m texture size)
        cum_dist = 0.0
        v_high = height / 10.0
        for i in range(n_base):
            p1 = coords[i]
            p2 = coords[(i+1)%n_base]
            
            # U = distance in meters / 10.0
            u = cum_dist / 10.0
            
            f.write(f"vt {u:.4f} 0.0\n")      # Bottom UV
            f.write(f"vt {u:.4f} {v_high:.4f}\n") # Top UV
            
            dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            cum_dist += dist

        # Add the closing UV pair for the last segment
        u_last = cum_dist / 10.0
        f.write(f"vt {u_last:.4f} 0.0\n")
        f.write(f"vt {u_last:.4f} {v_high:.4f}\n")
        
        # 3. Write Side Faces
        if mtl_filename:
            f.write("usemtl MatFacade\n")
        
        for i in range(n_base):
            v1 = i * 2 + 1
            v2 = i * 2 + 2
            v3 = ((i+1) % n_base) * 2 + 2
            v4 = ((i+1) % n_base) * 2 + 1
            
            # UV indices
            vt1 = v1
            vt2 = v2
            if i < n_base - 1:
                vt_bot_next = v4
                vt_top_next = v3
            else:
                vt_bot_next = n_base * 2 + 1
                vt_top_next = n_base * 2 + 2
            
            f.write(f"f {v1}/{vt1} {v2}/{vt2} {v3}/{vt_top_next} {v4}/{vt_bot_next}\n")

        # --- Vertices & UVs for Roof ---
        
        roof_idx_start = n_base * 2 + 1
        roof_uv_start = n_base * 2 + 3 # +2 for the extra closing vt pair
        
        # 4. Write Roof Vertices
        for vx, vy in roof_v2:
            f.write(f"v {vx:.4f} {vy:.4f} {height:.4f}\n")
            
        # 5. Write Roof UVs (10m x 10m scale)
        for vx, vy in roof_v2:
            f.write(f"vt {vx/10.0:.4f} {vy/10.0:.4f}\n")
            
        # 6. Write Roof Faces
        if mtl_filename:
            f.write("usemtl MatRoof\n")
            
        for face in roof_faces:
            v1_r = face[0] + roof_idx_start
            v2_r = face[1] + roof_idx_start
            v3_r = face[2] + roof_idx_start
            
            vt1_r = face[0] + roof_uv_start
            vt2_r = face[1] + roof_uv_start
            vt3_r = face[2] + roof_uv_start
            
            f.write(f"f {v1_r}/{vt1_r} {v2_r}/{vt2_r} {v3_r}/{vt3_r}\n")

    return True

def main():
    parser = argparse.ArgumentParser(description="Generate geospecific 3D models for buildings.")
    parser.add_argument("input_geojson", help="Path to input GeoJSON with footprints")
    parser.add_argument("output_dir", help="Directory to save OBJ models")
    parser.add_argument("output_csv", help="Path to save instances CSV (for placement)")
    parser.add_argument("--texture", help="Path to facade texture image")
    parser.add_argument("--roof_texture", help="Path to roof texture image")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_geojson):
        print(f"Error: Input file {args.input_geojson} not found.")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle Materials/Textures
    mtl_filename = None
    if args.texture:
        import shutil
        mtl_filename = "materials.mtl"
        mtl_path = os.path.join(args.output_dir, mtl_filename)
        
        tex_basename = os.path.basename(args.texture)
        shutil.copy2(args.texture, os.path.join(args.output_dir, tex_basename))
        
        roof_tex_basename = None
        if args.roof_texture and os.path.exists(args.roof_texture):
            roof_tex_basename = os.path.basename(args.roof_texture)
            shutil.copy2(args.roof_texture, os.path.join(args.output_dir, roof_tex_basename))
        
        with open(mtl_path, 'w') as f:
            f.write("newmtl MatFacade\nKd 1.0 1.0 1.0\nillum 2\n")
            f.write(f"map_Kd {tex_basename}\n\n")
            f.write("newmtl MatRoof\nKd 1.0 1.0 1.0\nillum 1\n")
            if roof_tex_basename:
                f.write(f"map_Kd {roof_tex_basename}\n")
            else:
                f.write("Kd 0.6 0.6 0.6\n")
    
    print(f"Loading {args.input_geojson}...")
    gdf = gpd.read_file(args.input_geojson)
    print(f"Loaded {len(gdf)} features.")
    
    instances = []
    
    # Check levels column
    levels_col = 'building:levels'
    if levels_col not in gdf.columns and 'building_l' in gdf.columns:
        levels_col = 'building_l'
        
    if levels_col not in gdf.columns:
        print(f"Warning: '{levels_col}' not found. Defaulting to 1 level.")
        gdf[levels_col] = 1
        
    # Ensure numeric
    gdf[levels_col] = pd.to_numeric(gdf[levels_col], errors='coerce').fillna(1)
    
    count_success = 0
    count_fail = 0
    
    for idx, row in gdf.iterrows():
        try:
            geom = row.geometry
            levels = row[levels_col]
            height = levels * 3.0 # As requested
            
            if geom is None or geom.is_empty:
                continue
                
            centroid = geom.centroid
            cx, cy = centroid.x, centroid.y
            
            shifted_geom = shapely.affinity.translate(geom, xoff=-cx, yoff=-cy)
            
            obj_name = f"geospecific_{idx}.obj"
            obj_path = os.path.join(args.output_dir, obj_name)
            
            success = write_textured_obj(obj_path, shifted_geom, height, mtl_filename)
            
            if not success:
                mesh = trimesh.creation.extrude_polygon(shifted_geom, height=height)
                if mesh:
                    mesh.export(obj_path, file_type='obj')
                else:
                    count_fail += 1
                    continue
            
            instances.append({
                'x': cx,
                'y': cy,
                'z': 0,
                'angle_deg': 0,
                'obj_filename': obj_name,
                'pred_width': 0,
                'pred_height': 0,
                'fit_dist': 0,
                'pred_levels': levels,
                'cluster_levels': levels
            })
            
            count_success += 1
            
            if idx % 100 == 0:
                print(f"Processed {idx} buildings...")
                
        except Exception as e:
            print(f"Error processing building {idx}: {e}")
            count_fail += 1
            
    print(f"Finished. Generated {count_success} models. Failed: {count_fail}")
    
    df_out = pd.DataFrame(instances)
    df_out.to_csv(args.output_csv, index=False)
    print(f"Saved {len(instances)} instances placement to {args.output_csv}")

if __name__ == "__main__":
    main()
