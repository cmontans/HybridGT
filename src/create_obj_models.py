import pandas as pd
import os
import argparse
import shutil

def write_material_file(filename, texture_filename, roof_texture_filename=None):
    """Writes the .mtl file."""
    with open(filename, 'w') as f:
        # Facade Material
        f.write("newmtl MatFacade\n")
        f.write("Ns 96.078431\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.500000 0.500000 0.500000\n")
        f.write("Ke 0.000000 0.000000 0.000000\n")
        f.write("Ni 1.000000\n")
        f.write("d 1.000000\n")
        f.write("illum 2\n")
        f.write(f"map_Kd {texture_filename}\n")
        f.write("\n")
        
        # Roof/Floor Material
        f.write("newmtl MatRoof\n")
        f.write("Ns 0.000000\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("Ke 0.000000 0.000000 0.000000\n")
        f.write("Ni 1.000000\n")
        f.write("d 1.000000\n")
        f.write("illum 1\n")
        if roof_texture_filename:
            f.write(f"map_Kd {roof_texture_filename}\n")
        else:
            f.write("Kd 0.600000 0.600000 0.600000\n") # Grey fallback

def write_box_obj(filename, width, height, building_height, mtl_filename=None):
    """
    Writes a simple box 3D model to an OBJ file.
    The box is centered on the XY plane and rests on Z=0.
    UVs are scaled to maintain aspect ratio (1 unit vertically = building height).
    """
    
    # Half dimensions
    hw = width / 2.0
    hh = height / 2.0
    
    # Vertices (Y-Up System)
    # X = Width
    # Y = Height (Up)
    # Z = Depth (Footprint Height)
    
    # Bottom face (Y=0)
    v1 = (-hw, 0, hh)
    v2 = (hw, 0, hh)
    v3 = (hw, 0, -hh)
    v4 = (-hw, 0, -hh)
    
    # Top face (Y=building_height)
    v5 = (-hw, building_height, hh)
    v6 = (hw, building_height, hh)
    v7 = (hw, building_height, -hh)
    v8 = (-hw, building_height, -hh)
    
    vertices = [v1, v2, v3, v4, v5, v6, v7, v8]
    
    with open(filename, 'w') as f:
        f.write(f"# Cluster Model: {width}m x {height}m x {building_height}m (Y-Up)\n")
        
        if mtl_filename:
            f.write(f"mtllib {mtl_filename}\n")
            
        f.write(f"o Building_{width}_{height}\n")
        
        # Write vertices (indices 1-8)
        for v in vertices:
            # OBJ is X Y Z
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            
        f.write("\n")
        
        if mtl_filename:
            # UV Scaling factors
            # U_max = Face_Width / Building_Height
            
            u_front = width / building_height
            u_side = height / building_height
            
            # Write UVs for each face
            
            # --- Front Face (Width) +Z ---
            # Using v1-v2-v6-v5
            f.write(f"vt 0.0 0.0\n")       # 1
            f.write(f"vt {u_front:.4f} 0.0\n") # 2
            f.write(f"vt {u_front:.4f} 1.0\n") # 3
            f.write(f"vt 0.0 1.0\n")       # 4
            
            # --- Right Face (Depth) -X? No, Width is X.
            # v2(hw,0,hh) to v3(hw,0,-hh) is -Z direction.
            # So Right Face is +X face: v2-v3-v7-v6
            f.write(f"vt 0.0 0.0\n")       # 5
            f.write(f"vt {u_side:.4f} 0.0\n")  # 6
            f.write(f"vt {u_side:.4f} 1.0\n")  # 7
            f.write(f"vt 0.0 1.0\n")       # 8
            
            # --- Back Face (Width) -Z ---
            # v3-v4-v8-v7
            f.write(f"vt 0.0 0.0\n")       # 9
            f.write(f"vt {u_front:.4f} 0.0\n") # 10
            f.write(f"vt {u_front:.4f} 1.0\n") # 11
            f.write(f"vt 0.0 1.0\n")       # 12
            
            # --- Left Face (Depth) -X ---
            # v4-v1-v5-v8
            f.write(f"vt 0.0 0.0\n")       # 13
            f.write(f"vt {u_side:.4f} 0.0\n")  # 14
            f.write(f"vt {u_side:.4f} 1.0\n")  # 15
            f.write(f"vt 0.0 1.0\n")       # 16
            
            # --- Top Face (Roof) ---
            # UVs scaled by building_height to match facade texture density
            f.write(f"vt 0.0 0.0\n")       # 17
            f.write(f"vt {width/building_height:.4f} 0.0\n") # 18
            f.write(f"vt {width/building_height:.4f} {height/building_height:.4f}\n") # 19
            f.write(f"vt 0.0 {height/building_height:.4f}\n") # 20

            f.write("usemtl MatFacade\n")
            
            # Front (1-2-6-5) -> UVs 1-2-3-4
            f.write("f 1/1 2/2 6/3 5/4\n")
            
            # Right (2-3-7-6) -> UVs 5-6-7-8
            f.write("f 2/5 3/6 7/7 6/8\n")
            
            # Back (3-4-8-7) -> UVs 9-10-11-12
            f.write("f 3/9 4/10 8/11 7/12\n")
            
            # Left (4-1-5-8) -> UVs 13-14-15-16
            f.write("f 4/13 1/14 5/15 8/16\n")
            
            f.write("usemtl MatRoof\n") # Explicit roof material
            # Bottom (1-4-3-2) - No UVs
            f.write("f 1 4 3 2\n")
            # Top (5-6-7-8) -> UVs 17-18-19-20
            f.write("f 5/17 6/18 7/19 8/20\n")
            
        else:
            # Simple faces without UVs
            f.write("f 1 4 3 2\n") # Bottom
            f.write("f 5 6 7 8\n") # Top
            f.write("f 1 2 6 5\n") # Front
            f.write("f 2 3 7 6\n") # Right
            f.write("f 3 4 8 7\n") # Back
            f.write("f 4 1 5 8\n") # Left

def main():
    parser = argparse.ArgumentParser(description="Generate 3D OBJ models for building clusters.")
    parser.add_argument("clusters_csv", help="Path to CSV containing cluster centers (width, height)")
    parser.add_argument("output_dir", help="Directory to save OBJ files")
    parser.add_argument("--texture", help="Path to facade texture image")
    parser.add_argument("--roof_texture", help="Path to roof texture image")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.clusters_csv):
        print(f"Error: CSV file not found at {args.clusters_csv}")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle Texture
    mtl_filename = None
    if args.texture:
        if not os.path.exists(args.texture):
            print(f"Error: Texture file not found at {args.texture}")
            return
            
        # Copy facade texture to output dir
        tex_basename = os.path.basename(args.texture)
        dest_tex_path = os.path.join(args.output_dir, tex_basename)
        shutil.copy2(args.texture, dest_tex_path)
        print(f"Copied facade texture to {dest_tex_path}")
        
        # Handle Roof Texture
        roof_tex_basename = None
        if args.roof_texture:
            if os.path.exists(args.roof_texture):
                roof_tex_basename = os.path.basename(args.roof_texture)
                dest_roof_tex_path = os.path.join(args.output_dir, roof_tex_basename)
                shutil.copy2(args.roof_texture, dest_roof_tex_path)
                print(f"Copied roof texture to {dest_roof_tex_path}")
            else:
                print(f"Warning: Roof texture not found at {args.roof_texture}")
            
        # Create materials.mtl
        mtl_filename = "materials.mtl"
        mtl_path = os.path.join(args.output_dir, mtl_filename)
        write_material_file(mtl_path, tex_basename, roof_tex_basename)
        print(f"Created material library at {mtl_path}")
    
    print(f"Loading clusters from {args.clusters_csv}...")
    try:
        df = pd.read_csv(args.clusters_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    # Check columns
    w_col = 'optimum_width'
    h_col = 'optimum_height'
    
    if w_col not in df.columns or h_col not in df.columns:
        print(f"Error: Expected columns '{w_col}' and '{h_col}'")
        print(f"Found: {df.columns.tolist()}")
        return
        
    has_levels = 'optimum_levels' in df.columns
        
    print(f"Generating models for {len(df)} clusters...")
    
    for i, row in df.iterrows():
        width = row[w_col]
        height = row[h_col]
        
        if width <= 0 or height <= 0:
            continue
            
        if has_levels:
            levels = int(row['optimum_levels'])
            building_height = levels * 3.0 # 3.0 meters per floor as requested
            filename = f"cluster_{i}_w{width:.2f}_h{height:.2f}_l{levels}.obj"
        else:
            building_height = 10.0 # Default fallback
            filename = f"cluster_{i}_w{width:.2f}_h{height:.2f}.obj"
            
        filepath = os.path.join(args.output_dir, filename)
        
        # Write OBJ
        write_box_obj(filepath, width, height, building_height, mtl_filename)
        
    print(f"Successfully generated {len(df)} OBJ models in '{args.output_dir}'")

if __name__ == "__main__":
    main()
