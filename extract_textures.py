"""Extract image textures from .blend files into per-asset folders.

Usage:
    blender -b --python extract_textures.py -- \
        --blend_dir /path/to/blends \
        --out_dir /path/to/animal_textures

Or specify individual files:
    blender -b --python extract_textures.py -- \
        --blend_files fur1.blend fur2.blend ... \
        --out_dir ./animal_textures
"""
import bpy
import os
import sys
import glob
import argparse

argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

p = argparse.ArgumentParser()
p.add_argument('--blend_files', nargs='+', default=[])
p.add_argument('--blend_dir', default=None,
               help='Directory of .blend files (will recurse)')
p.add_argument('--out_dir', required=True)
args = p.parse_args(argv)

# Collect blend files
blend_files = list(args.blend_files)
if args.blend_dir:
    blend_files += glob.glob(os.path.join(args.blend_dir, '**/*.blend'),
                             recursive=True)

if not blend_files:
    print("No .blend files found")
    sys.exit(1)


def detect_role(img):
    """Figure out if this image is color/normal/roughness/etc.

    First tries filename heuristics, then walks the shader node graph to see
    what Principled BSDF input the image connects to.
    """
    name = img.name.lower() + ' ' + (img.filepath or '').lower()
    if any(k in name for k in ['color', 'diff', 'albedo', 'basecolor', 'base_color', '_col.']):
        return 'color'
    if any(k in name for k in ['normalgl', 'normal_gl', 'normal', 'norm', '_nrm', '_n.']):
        return 'normal'
    if any(k in name for k in ['rough', '_r.']):
        return 'roughness'
    if 'metal' in name:
        return 'metallic'
    if any(k in name for k in ['ao', 'occlusion']):
        return 'ao'
    if any(k in name for k in ['displ', 'height', '_h.']):
        return 'displacement'

    # Fallback: walk the node graph
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type != 'TEX_IMAGE' or node.image != img:
                continue
            stack, visited = [node], set()
            while stack:
                cur = stack.pop()
                if id(cur) in visited:
                    continue
                visited.add(id(cur))
                for output in cur.outputs:
                    for link in output.links:
                        sock = link.to_socket.name.lower()
                        if 'base color' in sock or 'base_color' in sock:
                            return 'color'
                        if 'roughness' in sock:
                            return 'roughness'
                        if 'metallic' in sock:
                            return 'metallic'
                        if 'normal' in sock:
                            return 'normal'
                        # Walk through intermediate nodes
                        if link.to_node.type in ('NORMAL_MAP', 'MAPPING', 'BUMP',
                                                  'SEPRGB', 'SEP_COLOR', 'GAMMA',
                                                  'INVERT'):
                            stack.append(link.to_node)
    return None


def safe_name(s):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in s)


print(f"Processing {len(blend_files)} .blend files")
os.makedirs(args.out_dir, exist_ok=True)

for blend_path in blend_files:
    asset_name = safe_name(os.path.splitext(os.path.basename(blend_path))[0])
    asset_dir = os.path.join(args.out_dir, asset_name)

    print(f"\n=== {blend_path} ===")
    try:
        bpy.ops.wm.open_mainfile(filepath=blend_path)
    except Exception as e:
        print(f"  failed to open: {e}")
        continue

    os.makedirs(asset_dir, exist_ok=True)
    saved = 0
    for img in bpy.data.images:
        if img.name in ('Render Result', 'Viewer Node'):
            continue
        if not img.size or img.size[0] == 0 or img.size[1] == 0:
            continue

        role = detect_role(img)
        if role is None:
            print(f"  skip {img.name} (no recognized role)")
            continue

        out_path = os.path.join(asset_dir, f'{role}.png')
        # Disambiguate if same role found multiple times
        i = 2
        while os.path.exists(out_path):
            out_path = os.path.join(asset_dir, f'{role}_{i}.png')
            i += 1

        try:
            img.file_format = 'PNG'
            img.filepath_raw = out_path
            img.save()
            print(f"  saved {role}: {os.path.basename(out_path)} "
                  f"({img.size[0]}x{img.size[1]})")
            saved += 1
        except Exception as e:
            print(f"  failed {img.name}: {e}")

    print(f"  total saved: {saved}")

    # Empty asset_dir if nothing useful came out
    if saved == 0:
        try:
            os.rmdir(asset_dir)
        except OSError:
            pass

print("\nDone.")