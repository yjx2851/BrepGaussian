import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Ensure we can import json_to_surface_patches from the same directory
sys.path.append(os.path.dirname(__file__))

import json_to_surface_patches


def find_json_files(input_root: str) -> list:
    """
    Find all pipeline_results.json files under subdirectories of input_root.
    
    Returns:
        list: [(json_path, subdir_name), ...] tuples of (json_path, subdir_name)
    """
    json_files = []
    
    if not os.path.exists(input_root):
        print(f"Warning: input root directory does not exist: {input_root}")
        return json_files
    
    try:
        # Iterate over all subdirectories under input_root
        for subdir_name in os.listdir(input_root):
            subdir_path = os.path.join(input_root, subdir_name)
            
            # Only process directories
            if not os.path.isdir(subdir_path):
                continue
            
            # Look for pipeline_results.json
            json_path = os.path.join(subdir_path, "pipeline_results.json")
            if os.path.exists(json_path) and os.path.isfile(json_path):
                json_files.append((json_path, subdir_name))
    
    except Exception as e:
        print(f"错误: 遍历目录时出错: {e}")
    
    return sorted(json_files)


def process_single_json(json_path: str, output_dir: str):
    """
    Process a single JSON file and generate surface meshes.
    
    Args:
        json_path: Path to the JSON file.
        output_dir: Output directory where OBJ files will be written.
    """
    # Load JSON
    data = json_to_surface_patches.load_json(json_path)
    
    surfaces = data.get('surfaces', [])
    lines = data.get('lines', [])
    curves = data.get('curves', [])
    corners = data.get('corners', [])
    
    print(f"  Found {len(surfaces)} surfaces, {len(lines)} lines, {len(curves)} curves, {len(corners)} corners")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each surface
    all_meshes = []
    for i, surface in enumerate(surfaces):
        mesh = json_to_surface_patches.process_surface(
            surface, lines, curves, corners, output_dir, i
        )
        if mesh is not None:
            # Save individual surface OBJ
            obj_path = os.path.join(output_dir, f"surface_{i:03d}.obj")
            import open3d as o3d
            o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_normals=False)
            print(f"    Saved to: {obj_path}")
            
            all_meshes.append(mesh)
    
    # Merge all patches into a combined mesh
    if len(all_meshes) > 0:
        print(f"  Merging all surface patches...")
        combined_vertices = []
        combined_triangles = []
        vertex_offset = 0
        
        import open3d as o3d
        for mesh in all_meshes:
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            combined_vertices.append(vertices)
            combined_triangles.append(triangles + vertex_offset)
            vertex_offset += len(vertices)
        
        combined_mesh = o3d.geometry.TriangleMesh()
        combined_mesh.vertices = o3d.utility.Vector3dVector(np.vstack(combined_vertices))
        combined_mesh.triangles = o3d.utility.Vector3iVector(np.vstack(combined_triangles))
        combined_mesh.compute_vertex_normals()
        
        # Save combined OBJ
        combined_obj_path = os.path.join(output_dir, "all_surfaces.obj")
        o3d.io.write_triangle_mesh(combined_obj_path, combined_mesh, write_vertex_normals=False)
        print(f"  Combined OBJ saved to: {combined_obj_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch run json_to_surface_patches over multiple JSON files')
    parser.add_argument('--inputs', type=str, default='./line_batch_output_merge_aligned',
                        help='Input root directory')
    parser.add_argument('--output', type=str, default='surface_batch_output_merge_aligned',
                        help='Output root directory')
    parser.add_argument('--force', action='store_true',
                        help='Force re-processing even if output directory already exists')
    
    args = parser.parse_args()
    
    input_root = args.inputs
    output_root = args.output
    # By default, skip existing outputs unless --force is provided
    skip_existing = not args.force
    
    # Find all JSON files
    json_files = find_json_files(input_root)
    if not json_files:
        print(f"No pipeline_results.json files found under input root: {input_root}")
        print(f"  Please ensure directory structure is: {input_root}\\<subdir>\\pipeline_results.json")
        return
    
    print('=' * 60)
    print(f"Batch start: {len(json_files)} JSON files")
    print(f"  Input root: {input_root}")
    print(f"  Output root: {output_root}")
    print(f"  Skip existing: {'yes' if skip_existing else 'no'}")
    print('=' * 60)
    
    success = 0
    fail = 0
    skipped = 0
    
    for idx, (json_path, subdir_name) in enumerate(json_files, 1):
        out_dir = os.path.join(output_root, subdir_name)
        print(f"\n[{idx}/{len(json_files)}] Processing: {json_path}")
        print(f"  Output directory: {out_dir}")
        
        # Check if output directory already has generated meshes
        if skip_existing and os.path.exists(out_dir) and os.path.isdir(out_dir):
            # Look for at least one surface_*.obj or all_surfaces.obj
            has_output = False
            try:
                if os.path.exists(os.path.join(out_dir, "all_surfaces.obj")):
                    has_output = True
                else:
                    # 检查是否有 surface_*.obj 文件
                    for fname in os.listdir(out_dir):
                        if fname.startswith("surface_") and fname.endswith(".obj"):
                            has_output = True
                            break
            except Exception:
                pass
            
            if has_output:
                skipped += 1
                print(f"  ⊘ Skipped (output directory already contains meshes)")
                continue
        
        try:
            process_single_json(json_path, out_dir)
            success += 1
            print(f"  ✓ Success")
        except Exception as e:
            fail += 1
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print('\n' + '=' * 60)
    print(f"Batch finished: success={success}, failed={fail}, skipped={skipped}")
    print(f"  Output root: {output_root}")
    print('=' * 60)


if __name__ == '__main__':
    main()

