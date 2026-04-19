import os
import sys
import argparse
from datetime import datetime

# Ensure we can import surface_fitting_pipeline from the same directory
sys.path.append(os.path.dirname(__file__))

from surface_fitting_pipeline import run_pipeline


def find_pcd_files(inputs_dir: str) -> list:
    files = []
    try:
        for name in os.listdir(inputs_dir):
            if name.lower().endswith('.pcd'):
                files.append(os.path.join(inputs_dir, name))
    except Exception:
        pass
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description='Batch run surface_fitting_pipeline on multiple PCD files')
    parser.add_argument('--inputs', type=str, default=r'./inputs_merge_aligned',
                        help='Input PCD directory')
    parser.add_argument('--output', type=str, default='line_batch_output_merge_aligned',
                        help='Output root directory')
    parser.add_argument('--plane_threshold', type=float, default=0.01)
    parser.add_argument('--cylinder_threshold', type=float, default=0.01)
    parser.add_argument('--min_points', type=int, default=50)
    # In batch mode, visualization is disabled by default (JSON/OBJ are still saved)
    parser.add_argument('--visualize', action='store_true',
                        help='Enable Open3D visualization windows (disabled by default)')

    args = parser.parse_args()

    inputs_dir = args.inputs
    output_root = args.output
    os.makedirs(output_root, exist_ok=True)

    pcd_files = find_pcd_files(inputs_dir)
    if not pcd_files:
        print(f"No PCD files found in input directory: {inputs_dir}")
        return

    print('=' * 60)
    print(f"Batch start: {len(pcd_files)} files, input={inputs_dir}, output_root={output_root}")
    print('=' * 60)

    success = 0
    fail = 0
    for idx, pcd_path in enumerate(pcd_files, 1):
        base = os.path.splitext(os.path.basename(pcd_path))[0]
        out_dir = os.path.join(output_root, base)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[{idx}/{len(pcd_files)}] Processing: {pcd_path}")
        try:
            run_pipeline(
                pcd_path=pcd_path,
                output_dir=out_dir,
                plane_threshold=args.plane_threshold,
                cylinder_threshold=args.cylinder_threshold,
                min_points=args.min_points,
                visualize=args.visualize
            )
            success += 1
        except Exception as e:
            fail += 1
            print(f"  Error: {e}")

    print('\n' + '=' * 60)
    print(f"Batch finished: success={success}, failed={fail}. Output root: {output_root}")
    print('=' * 60)


if __name__ == '__main__':
    main()


