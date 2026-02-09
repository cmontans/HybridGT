import pandas as pd
import os
import argparse
import shutil

# Constants
BUILDING_HEIGHT = 10.0
PARAPET_HEIGHT = 0.5
PLINTH_HEIGHT = 1.0
PLINTH_OFFSET = 0.2 # Extrusion

def write_material_file(filename, texture_filename):
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
        
        # Roof/Floor Material (Simple Grey)
        f.write("newmtl MatRoof\n")
        f.write("Ns 0.000000\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 0.600000 0.600000 0.600000\n") # Grey
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("Ke 0.000000 0.000000 0.000000\n")
        f.write("Ni 1.000000\n")
        f.write("d 1.000000\n")
        f.write("illum 1\n")

def write_obj(filename, vertices, faces, uvs=None, mtl_filename=None, obj_name="Building"):
    """
    Generic OBJ writer.
    vertices: list of (x, y, z) tuples
    faces: list of tuples (material_name, [v_idx1, v_idx2, ...], [uv_idx1, ...])
    uvs: list of (u, v) tuples
    """
    with open(filename, 'w') as f:
        f.write(f"# Generated LOD Model\n")
        if mtl_filename:
            f.write(f"mtllib {mtl_filename}\n")
        f.write(f"o {obj_name}\n")
        
        # Vertices
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            
        # UVs
        if uvs:
            for uv in uvs:
                f.write(f"vt {uv[0]:.4f} {uv[1]:.4f}\n")
                
        f.write("\n")
        
        # Faces
        current_mat = None
        for mat_name, v_indices, uv_indices in faces:
            if mat_name != current_mat:
                if mtl_filename:
                    f.write(f"usemtl {mat_name}\n")
                current_mat = mat_name
            
            f.write("f")
            for i in range(len(v_indices)):
                v_idx = v_indices[i]
                if uv_indices:
                    vt_idx = uv_indices[i]
                    f.write(f" {v_idx}/{vt_idx}")
                else:
                    f.write(f" {v_idx}")
            f.write("\n")

def generate_lod2(width, depth):
    """LOD2: Scan (Low). Simple Box."""
    hw = width / 2.0
    hd = depth / 2.0
    h = BUILDING_HEIGHT
    
    # Vertices (Y-Up: X=Width, Y=Height, Z=Depth)
    verts = [
        (-hw, 0, hd), (hw, 0, hd), (hw, 0, -hd), (-hw, 0, -hd), # Bottom 1-4
        (-hw, h, hd), (hw, h, hd), (hw, h, -hd), (-hw, h, -hd)  # Top 5-8
    ]
    
    # UVs
    u_front = width / h
    u_side = depth / h
    uvs = [
        (0,0), (u_front,0), (u_front,1), (0,1), # Front
        (0,0), (u_side,0), (u_side,1), (0,1)    # Right/Left/Back (reused logic)
    ]
    
    # Faces: (Material, [Verts 1-based], [UVs 1-based])
    faces = [
        # Walls
        ('MatFacade', [1, 2, 6, 5], [1, 2, 3, 4]), # Front
        ('MatFacade', [2, 3, 7, 6], [5, 6, 7, 8]), # Right
        ('MatFacade', [3, 4, 8, 7], [1, 2, 3, 4]), # Back
        ('MatFacade', [4, 1, 5, 8], [5, 6, 7, 8]), # Left
        # Roof/Floor
        ('MatRoof', [5, 6, 7, 8], []), # Top
        ('MatRoof', [1, 4, 3, 2], [])  # Bottom
    ]
    
    return verts, faces, uvs

def generate_lod1(width, depth):
    """LOD1: Med. Box + Parapet (Recessed Roof)."""
    hw = width / 2.0
    hd = depth / 2.0
    h = BUILDING_HEIGHT
    h_roof = h - PARAPET_HEIGHT
    
    # Vertices
    verts = [
        (-hw, 0, hd), (hw, 0, hd), (hw, 0, -hd), (-hw, 0, -hd), # 1-4 Bottom
        (-hw, h, hd), (hw, h, hd), (hw, h, -hd), (-hw, h, -hd), # 5-8 Top Rim (Outer)
        # Inner Roof Floor (Recessed) - Assume 0.3m wall thickness?
        # Let's just make it simple: Vertices at wall locations but lower height?
        # No, Parapet has thickness. Let's assume zero thickness for simplicity, or 0.2m.
        # Let's do 0.2m thickness.
    ]
    
    t = 0.2 # Wall thickness
    verts.extend([
        (-hw+t, h_roof, hd-t), (hw-t, h_roof, hd-t), (hw-t, h_roof, -hd+t), (-hw+t, h_roof, -hd+t), # 9-12 Roof Floor
        (-hw+t, h, hd-t), (hw-t, h, hd-t), (hw-t, h, -hd+t), (-hw+t, h, -hd+t) # 13-16 Top Rim (Inner)
    ])
    
    # UVs
    u_front = width / h
    u_side = depth / h
    uvs = [
        (0,0), (u_front,0), (u_front,1), (0,1), # Front/Back
        (0,0), (u_side,0), (u_side,1), (0,1)    # Sides
    ]
    
    faces = [
        # Outer Walls
        ('MatFacade', [1, 2, 6, 5], [1, 2, 3, 4]), # Front
        ('MatFacade', [2, 3, 7, 6], [5, 6, 7, 8]), # Right
        ('MatFacade', [3, 4, 8, 7], [1, 2, 3, 4]), # Back
        ('MatFacade', [4, 1, 5, 8], [5, 6, 7, 8]), # Left
        
        # Parapet Top (Rim) - Connect Outer Top (5-8) to Inner Top (13-16)
        ('MatRoof', [5, 6, 14, 13], []),
        ('MatRoof', [6, 7, 15, 14], []),
        ('MatRoof', [7, 8, 16, 15], []),
        ('MatRoof', [8, 5, 13, 16], []),
        
        # Parapet Inner Walls - Connect Inner Top (13-16) to Roof Floor (9-12)
        ('MatRoof', [13, 14, 10, 9], []),
        ('MatRoof', [14, 15, 11, 10], []),
        ('MatRoof', [15, 16, 12, 11], []),
        ('MatRoof', [16, 13, 9, 12], []),
        
        # Roof Floor
        ('MatRoof', [9, 10, 11, 12], []),
        
        # Bottom
        ('MatRoof', [1, 4, 3, 2], [])
    ]
    
    return verts, faces, uvs

def generate_lod0(width, depth):
    """LOD0: High. Box + Parapet + Plinth (Base)."""
    # Reuse LOD1 logic but split the outer walls
    hw = width / 2.0
    hd = depth / 2.0
    h = BUILDING_HEIGHT
    h_roof = h - PARAPET_HEIGHT
    h_plinth = PLINTH_HEIGHT
    off = PLINTH_OFFSET
    t = 0.2
    
    # Vertices
    # 1-4: Bottom (Expanded by offset)
    verts = [
        (-hw-off, 0, hd+off), (hw+off, 0, hd+off), (hw+off, 0, -hd-off), (-hw-off, 0, -hd-off)
    ]
    # 5-8: Plinth Top (Expanded)
    verts.extend([
        (-hw-off, h_plinth, hd+off), (hw+off, h_plinth, hd+off), (hw+off, h_plinth, -hd-off), (-hw-off, h_plinth, -hd-off)
    ])
    # 9-12: Main Wall Bottom (Standard Width) - Plinth Top Inner?
    # Simple Ledge: Plinth Top goes IN to Main Wall.
    verts.extend([
        (-hw, h_plinth, hd), (hw, h_plinth, hd), (hw, h_plinth, -hd), (-hw, h_plinth, -hd)
    ])
    # 13-16: Main Wall Top
    verts.extend([
        (-hw, h, hd), (hw, h, hd), (hw, h, -hd), (-hw, h, -hd)
    ])
    # 17-20: Roof Floor
    verts.extend([
        (-hw+t, h_roof, hd-t), (hw-t, h_roof, hd-t), (hw-t, h_roof, -hd+t), (-hw+t, h_roof, -hd+t)
    ])
    # 21-24: Inner Top Rim
    verts.extend([
        (-hw+t, h, hd-t), (hw-t, h, hd-t), (hw-t, h, -hd+t), (-hw+t, h, -hd+t)
    ])
    
    # UVs
    u_front = width / h
    u_side = depth / h
    uvs = [
        (0,0), (u_front,0), (u_front,1), (0,1), # Full Main
        (0,0), (u_side,0), (u_side,1), (0,1) 
    ]
    
    faces = [
        # Plinth Faces (Expanded)
        ('MatRoof', [1, 2, 6, 5], []),
        ('MatRoof', [2, 3, 7, 6], []),
        ('MatRoof', [3, 4, 8, 7], []),
        ('MatRoof', [4, 1, 5, 8], []),
        
        # Plinth Ledge (Top of plinth to start of wall)
        ('MatRoof', [5, 6, 10, 9], []),
        ('MatRoof', [6, 7, 11, 10], []),
        ('MatRoof', [7, 8, 12, 11], []),
        ('MatRoof', [8, 5, 9, 12], []),
        
        # Main Walls (From 9-12 to 13-16)
        ('MatFacade', [9, 10, 14, 13], [1, 2, 3, 4]),
        ('MatFacade', [10, 11, 15, 14], [5, 6, 7, 8]),
        ('MatFacade', [11, 12, 16, 15], [1, 2, 3, 4]),
        ('MatFacade', [12, 9, 13, 16], [5, 6, 7, 8]),
        
        # Parapet (Rim)
        ('MatRoof', [13, 14, 22, 21], []),
        ('MatRoof', [14, 15, 23, 22], []),
        ('MatRoof', [15, 16, 24, 23], []),
        ('MatRoof', [16, 13, 21, 24], []),
        
        # Parapet Inner
        ('MatRoof', [21, 22, 18, 17], []),
        ('MatRoof', [22, 23, 19, 18], []),
        ('MatRoof', [23, 24, 20, 19], []),
        ('MatRoof', [24, 21, 17, 20], []),
        
        # Roof Floor
        ('MatRoof', [17, 18, 19, 20], []),
        
        # Bottom
        ('MatRoof', [1, 4, 3, 2], [])
    ]
    
    return verts, faces, uvs

def main():
    parser = argparse.ArgumentParser(description="Generate LOD models for last cluster.")
    parser.add_argument("clusters_csv", help="Clusters CSV")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--texture", help="Texture file")
    
    args = parser.parse_args()
    
    # Read last cluster
    df = pd.read_csv(args.clusters_csv)
    last_row = df.iloc[-1]
    
    idx = df.index[-1]
    width = last_row['optimum_width']
    depth = last_row['optimum_height'] # Treat as depth
    
    print(f"Generating LODs for Cluster {idx}: W={width}, D={depth}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle Texture
    mtl_filename = "materials.mtl"
    if args.texture:
        shutil.copy2(args.texture, os.path.join(args.output_dir, os.path.basename(args.texture)))
        write_material_file(os.path.join(args.output_dir, mtl_filename), os.path.basename(args.texture))
    
    # LOD2 (Low)
    verts, faces, uvs = generate_lod2(width, depth)
    write_obj(os.path.join(args.output_dir, f"cluster_{idx}_lod2.obj"), verts, faces, uvs, mtl_filename, f"Building_LOD2_{idx}")
    
    # LOD1 (Med)
    verts, faces, uvs = generate_lod1(width, depth)
    write_obj(os.path.join(args.output_dir, f"cluster_{idx}_lod1.obj"), verts, faces, uvs, mtl_filename, f"Building_LOD1_{idx}")
    
    # LOD0 (High)
    verts, faces, uvs = generate_lod0(width, depth)
    write_obj(os.path.join(args.output_dir, f"cluster_{idx}_lod0.obj"), verts, faces, uvs, mtl_filename, f"Building_LOD0_{idx}")
    
    print("Done.")

if __name__ == "__main__":
    main()
