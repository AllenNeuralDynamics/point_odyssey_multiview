import numpy as np
import cv2
import os
import time
import glob
import matplotlib
from utils.file_io import read_tiff, write_png
from tqdm import tqdm
import json
import utils.plotting as plotting
import shutil


def _normalize_key(s):
    return s.lower().split()[0].split('_')[0]


def read_obj_file(obj_path:str):
    '''
        Load .obj file, return vertices, faces.
        return: vertices: N_v X 3, faces: N_f X 3
        '''
    obj_f = open(obj_path, 'r')
    lines = obj_f.readlines()
    vertices = []
    faces = []
    for ori_line in lines:
        line = ori_line.split()
        if line[0] == 'v':
            vertices.append([float(line[1]), float(line[2]), float(line[3])])  # x, y, z
        elif line[0] == 'f':  # Need to consider / case, // case, etc.
            faces.append([int(line[3].split('/')[0]),
                          int(line[2].split('/')[0]),
                          int(line[1].split('/')[0]) \
                          ])  # Notice! Need to reverse back when using the face since here it would be clock-wise!
            # Convert face order from clockwise to counter-clockwise direction.
    obj_f.close()

    return np.asarray(vertices), np.asarray(faces)


def reprojection(points: np.ndarray, K: np.ndarray, RT: np.ndarray, h: int, w: int):
    v = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    XYZ = (RT @ v.T).T[:, :3]
    Z = XYZ[:, 2:]
    XYZ = XYZ / XYZ[:, 2:]
    xyz = (K @ XYZ.T).T
    uv = xyz[:, :2]
    return uv, Z


def farthest_point_sampling(p, K):
    """
    greedy farthest point sampling
    p: point cloud
    K: number of points to sample
    """

    farthest_point = np.zeros((K, 3))
    idx = []
    max_idx = np.random.randint(0, p.shape[0] -1)
    farthest_point[0] = p[max_idx]
    idx.append(max_idx)
    print('farthest point sampling')
    for i in range(1, K):
        pairwise_distance = np.linalg.norm(p[:, None, :] - farthest_point[None, :i, :], axis=2)
        distance = np.min(pairwise_distance, axis=1, keepdims=True)
        max_idx = np.argmax(distance)
        farthest_point[i] = p[max_idx]
        idx.append(max_idx)
    print('farthest point sampling done')
    return farthest_point, idx


def valid_candidate_indices(asset_idx, asset_name, tracking_num,
                            obj_root, exr_root, cam_root, frames,
                            pallette, R1, R2, R3):
    """Mesh vertex indices that are on-screen and unoccluded for one asset
    as seen from a single view.

    This is the per-view candidate pool used by the multiview shared selection.
    The logic mirrors the single-view selection in __main__: it scans early
    frames, keeps the frame with the most valid vertices, and returns that
    frame's valid vertex indices (without sampling, so callers can union the
    pool across views before sampling the requested number of points).
    """
    best_valid = np.array([], dtype=np.int64)
    max_points = 0
    for i in range(0, len(frames) - 2):
        if not os.path.exists(os.path.join(obj_root, '{}_{}.obj'.format(asset_name, str(i + 2).zfill(4)))):
            continue
        initial_depth = read_tiff(os.path.join(exr_root, 'depth_{}.tiff'.format(str(i + 1).zfill(5))))
        initial_mask = cv2.imread(os.path.join(exr_root, 'segmentation_{}.png'.format(str(i + 1).zfill(5))))

        h, w, _ = initial_depth.shape

        inital_points, _ = read_obj_file(os.path.join(obj_root, '{}_{}.obj'.format(asset_name, str(i + 1).zfill(4))))
        inital_K = np.loadtxt(os.path.join(cam_root, 'K_{}.txt'.format(str(i + 1).zfill(4))))
        inital_RT = np.loadtxt(os.path.join(cam_root, 'RT_{}.txt'.format(str(i + 1).zfill(4))))
        inital_RT = R3 @ R2 @ inital_RT @ R1

        uv, z = reprojection(inital_points, inital_K, inital_RT, h, w)
        uv = np.round(uv).astype(np.int32)

        asset_mask = np.logical_and(initial_mask[:, :, 2] == pallette[asset_idx + 1, 0],
                                    initial_mask[:, :, 1] == pallette[asset_idx + 1, 1])
        asset_mask = np.logical_and(asset_mask, initial_mask[:, :, 0] == pallette[asset_idx + 1, 2])

        uv_mask_0 = np.logical_and(np.logical_and(uv[:, 0] >= 2, uv[:, 0] < w - 3),
                                   np.logical_and(uv[:, 1] >= 2, uv[:, 1] < h - 3))[:, None]
        uv_idx_0 = np.where(uv_mask_0)[0]
        if len(uv_idx_0) == 0:
            continue
        uv = uv[uv_idx_0]
        uv_mask = z[uv_idx_0] <= initial_depth[uv[:, 1], uv[:, 0]]

        # to make sure points are not on the boundary
        for delta_u in range(-2, 3):
            for delta_v in range(-2, 3):
                uv_mask = np.logical_and(uv_mask, 0.995 * z[uv_idx_0] <= initial_depth[uv[:, 1] + delta_v, uv[:, 0] + delta_u])

        uv_idx = np.where(uv_mask)[0]

        if len(uv_idx) and max_points < len(uv_idx):
            max_points = len(uv_idx)
            best_valid = uv_idx_0[uv_idx]
        if (len(uv_idx) >= 0.05 * len(inital_points) and i > len(frames) // 3) or len(uv_idx) > tracking_num:
            break
    return best_valid


def tracking(cp_root: str, data_root: str, tracking_index_list: list,
             pallette: np.array, sampling_scene_num=1000, depth_max=500, 
             scale=1, temporal_window=10, frame_num=None,
             obj_root=None, cam_root=None, tracking_points=None):
    
    obj_root = obj_root if obj_root else os.path.join(data_root, 'obj')
    img_root = os.path.join(cp_root, 'images')
    exr_root = os.path.join(cp_root, 'exr_img')
    cam_root = cam_root if cam_root else os.path.join(cp_root, 'cam')
    
    if os.path.exists(os.path.join(cp_root, 'obj')):
        obj_root = os.path.join(cp_root, 'obj')

    save_rgbs_root = os.path.join(data_root, 'rgbs')
    save_depths_root = os.path.join(data_root, 'depths')
    save_masks_root = os.path.join(data_root, 'masks')
    save_normals_root = os.path.join(data_root, 'normals')

    os.makedirs(save_rgbs_root, exist_ok=True)
    os.makedirs(save_depths_root, exist_ok=True)
    os.makedirs(save_masks_root, exist_ok=True)
    os.makedirs(save_normals_root, exist_ok=True)

    tracking_results = []
    tracking_results_3d = []
    K_data = []
    RT_data = []
    frames = sorted(glob.glob(os.path.join(img_root, '*.png')))

    search_list = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])

    # When a shared ground-point set is supplied (multiview), reuse it verbatim so
    # every view tracks the same scene points; otherwise sample them here as before.
    if tracking_points is None:
        tracking_points = np.random.uniform(-4, 4, (sampling_scene_num, 3))
        tracking_points[:, 1] *= 0
        tracking_points *= scale

    for i in tqdm(range(0, len(frames) - 1)) if frame_num is None else tqdm(range(0, frame_num)):
        tracking_results.append([])
        tracking_results_3d.append([])
        K = np.loadtxt(os.path.join(cam_root, 'K_{}.txt'.format(str(i + 1).zfill(4))))
        RT = np.loadtxt(os.path.join(cam_root, 'RT_{}.txt'.format(str(i + 1).zfill(4))))
        RT = R3 @ R2 @ RT @ R1
        depth = read_tiff(os.path.join(exr_root, 'depth_{}.tiff'.format(str(i + 1).zfill(5))))
        mask = cv2.imread(os.path.join(exr_root, 'segmentation_{}.png'.format(str(i + 1).zfill(5))))
        img = cv2.imread(os.path.join(exr_root, 'rgb_{}.png'.format(str(i + 1).zfill(5))))

        # convert img to jpg
        save_img_path = os.path.join(save_rgbs_root, 'rgb_{}.jpg'.format(str(i).zfill(5)))
        cv2.imwrite(save_img_path, img)

        # convert depth to 16 bit png
        save_depth_path = os.path.join(save_depths_root, 'depth_{}.png'.format(str(i).zfill(5)))
        max_value = 1000
        min_value = 0
        data = depth.copy()
        data[data > max_value] = max_value
        data[data < min_value] = min_value
        data = (data - min_value) * 65535 / (max_value - min_value)
        data = data.astype(np.uint16)
        write_png(data, save_depth_path)

        # cp normals and masks
        save_normal_path = os.path.join(save_normals_root, 'normal_{}.jpg'.format(str(i).zfill(5)))
        save_mask_path = os.path.join(save_masks_root, 'mask_{}.png'.format(str(i).zfill(5)))
        save_normal = cv2.imread(os.path.join(exr_root, 'normal_{}.png'.format(str(i + 1).zfill(5))))
        cv2.imwrite(save_normal_path, save_normal)

        if os.path.exists(os.path.join(exr_root, 'segmentation_{}.png'.format(str(i + 1).zfill(5)))):
            shutil.copy(os.path.join(exr_root, 'segmentation_{}.png'.format(str(i + 1).zfill(5))), save_mask_path)

        h, w, _ = img.shape
        for idx in range(len(assets)):
            asset = assets[idx]
            tracking_idx = tracking_index_list[idx]
            if not len(tracking_idx):
                continue
            asset_name = asset.replace('.', '_')
            obj_path = os.path.join(obj_root, '{}_{}.obj'.format(asset_name, str(i + 2).zfill(4)))
            obj_v, obj_f = read_obj_file(obj_path)
            obj_v = obj_v[tracking_idx]
            uv, z = reprojection(obj_v, K, RT, h, w)
            visibility = np.zeros((uv.shape[0], 1))
            # find nearest depth
            asset_mask = np.logical_and(mask[:, :, 2] == pallette[idx + 1, 0],
                                        mask[:, :, 1] == pallette[idx + 1, 1])
            asset_mask = np.logical_and(asset_mask, mask[:, :, 0] == pallette[idx + 1, 2])

            for j in range(len(uv)):
                u, v = uv[j]
                if u < 0 or u >= w or v < 0 or v >= h:
                    visibility[j] = 0
                    continue
                else:
                    for delta_uv in search_list:
                        u_, v_ = np.floor(uv[j]).astype(np.int32) + delta_uv
                        if u_ < 0 or u_ >= w or v_ < 0 or v_ >= h:
                            continue
                        if asset_mask[int(v_), int(u_)]:
                            if not asset_mask[int(v), int(u)]:
                                uv[j] = np.array([u_, v_])
                                break

                    v_low = np.floor(uv[j, 1]).astype(np.int32)
                    v_high = np.min([np.ceil(uv[j, 1]).astype(np.int32), h - 1])
                    u_low = np.floor(uv[j, 0]).astype(np.int32)
                    u_high = np.min([np.ceil(uv[j, 0]).astype(np.int32), w - 1])
                    # find nearest depth
                    d_max = np.max(depth[v_low:v_high + 1, u_low:u_high + 1])
                    d_median = np.median(depth[v_low:v_high + 1, u_low:u_high + 1])
                    if z[j] < 0 or z[j] > depth_max:
                        visibility[j] = 2
                        # print('invalid depth')
                        continue
                    if d_max >= 0.97 * z[j] and z[j] > 0.95 * d_median and z[j] < 1.05 * d_median: #and d_max - z[j] * 0.99 < d_threshold * scale:
                        visibility[j] = 1

            tracking_results[i].append(np.concatenate((uv, visibility), axis=1))
            tracking_results_3d[i].append(obj_v)

        # track ground points
        uv, z = reprojection(tracking_points, K, RT, h, w)
        visibility = np.zeros((uv.shape[0], 1))
        # find nearest depth
        asset_mask = np.logical_and(mask[:, :, 2] == 0,
                                    mask[:, :, 1] == 0)
        asset_mask = np.logical_and(asset_mask, mask[:, :, 0] == 0)

        for j in range(len(uv)):
            u, v = uv[j]
            if u < 0 or u >= w or v < 0 or v >= h:
                visibility[j] = 0
                continue
            else:
                ## sampling on the neighborhood
                for delta_uv in search_list:
                    u_, v_ = np.floor(uv[j]).astype(np.int32) + delta_uv
                    if u_ < 0 or u_ >= w or v_ < 0 or v_ >= h:
                        continue
                    if asset_mask[int(v_), int(u_)]:
                        if not asset_mask[int(v), int(u)]:
                            uv[j] = np.array([u_, v_])
                            break
                v_low = np.floor(uv[j, 1]).astype(np.int32)
                v_high = np.min([np.ceil(uv[j, 1]).astype(np.int32), h - 1])
                u_low = np.floor(uv[j, 0]).astype(np.int32)
                u_high = np.min([np.ceil(uv[j, 0]).astype(np.int32), w - 1])
                # find nearest depth
                d_max = np.max(depth[v_low:v_high + 1, u_low:u_high + 1])
                if z[j] < 0 or z[j] > depth_max:
                    visibility[j] = 2
                    continue
                if d_max >= 0.97 * z[j]:
                    visibility[j] = 1
        tracking_results[i].append(np.concatenate((uv, visibility), axis=1))
        tracking_results_3d[i].append(tracking_points)

        tracking_results[i] = np.concatenate(tracking_results[i], axis=0)
        tracking_results_3d[i] = np.concatenate(tracking_results_3d[i], axis=0)

        K_data.append(K.astype(np.float16))
        RT_data.append(RT.astype(np.float16))
    tracking_results = np.stack(tracking_results, axis=0)
    tracking_results = tracking_results.astype(np.float16)

    tracking_results_3d = np.stack(tracking_results_3d, axis=0)
    tracking_results_3d = tracking_results_3d.astype(np.float16)

    K_data = np.stack(K_data, axis=0)
    K_data = K_data.astype(np.float16)

    RT_data = np.stack(RT_data, axis=0)
    RT_data = RT_data.astype(np.float16)

    # save annotations as npz

    np.savez(os.path.join(data_root, 'annotations.npz'),
             trajs_2d=tracking_results[:, :, :2],
             trajs_3d=tracking_results_3d,
             visibilities=tracking_results[:, :, 2],
             intrinsics=K_data,
             extrinsics=RT_data)

    return tracking_results



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./results/robot_dome')
    parser.add_argument('--cp_root', type=str, default='./results/robot_dome')
    parser.add_argument('--sampling_points', type=int, default=5000)
    parser.add_argument('--sampling_scene_points', type=int, default=1000)
    parser.add_argument('--depth_max', type=float, default=500)
    parser.add_argument('--scale', type=float, default=10)

    parser.add_argument('--save_num', type=int, default=None)
    parser.add_argument('--temporal_window', type=int, default=10)
    parser.add_argument('--farthest_sampling', action='store_true', default=False)

    parser.add_argument('--obj_root', type=str, default=None,
                        help='Override path to obj directory (default: cp_root/obj)')
    parser.add_argument('--cam_root', type=str, default=None,
                        help='Override path to cam directory (default: cp_root/cam)')
    parser.add_argument('--scene_root', type=str, default=None,
                        help='Override path to scene_info.json directory (default: cp_root)')

    # tracking RNG seed: controls vertex sampling + ground-point locations.
    # Separate from the render seed (scene content). Unset -> time+pid (varies
    # per run); set -> reproducible.
    parser.add_argument('--seed', type=int, default=None,
                        help='Tracking RNG seed (vertex sampling + ground points). '
                             'Unset -> time+pid (varies per run); set -> reproducible.')

    # multiview unified tracking: select ONE shared set of points (visible in
    # >=1 view) and track it identically from every view, instead of selecting
    # a different set per view. The view layout is derived from --scene_root
    # (the multiview output dir holding view*/ and cam/view*/).
    parser.add_argument('--unified_tracking', action='store_true', default=False,
                        help='Select a single point set shared across all views '
                             '(union of per-view visible vertices) instead of a '
                             'separate per-view set. Views are auto-detected under '
                             '--scene_root.')

    args = parser.parse_args()

    # Resolve the tracking seed (mirrors the render seed convention).
    if args.seed is not None:
        tracking_seed = args.seed
    else:
        tracking_seed = (int(time.time() * 1e6) + os.getpid()) % (2 ** 32)
    print('tracking seed: {}'.format(tracking_seed))
    np.random.seed(tracking_seed)

    exr_root = os.path.join(args.cp_root, 'exr_img')
    obj_root = args.obj_root if args.obj_root else os.path.join(args.cp_root, 'obj')
    cam_root = args.cam_root if args.cam_root else os.path.join(args.cp_root, 'cam')

    scene_root = args.scene_root if args.scene_root else args.cp_root
    scene_info = json.load(open(os.path.join(scene_root, 'scene_info.json'), 'r'))

    assets = scene_info['assets']
    character_name = scene_info['character'] if 'character' in scene_info.keys() else None
    # character_assets = [i for i in assets if character_name in i] if character_name else []
    if 'character_meshes' in scene_info and scene_info['character_meshes']:
        character_assets = scene_info['character_meshes']
    elif character_name:
        char_key = _normalize_key(character_name)
        character_assets = [i for i in assets if _normalize_key(i).startswith(char_key)]
    else:
        character_assets = []


    obj_assets = [i for i in assets if i not in character_assets and i != 'background']
    # character_sampling_num = args.sampling_points // len(character_assets) if character_name else 0
    character_sampling_num = args.sampling_points // len(character_assets) if (character_name and len(character_assets) > 0) else 0
    obj_sampling_num = args.sampling_points // len(obj_assets)
    print('character sampling num: {}, object sampling num: {}'.format(character_sampling_num, obj_sampling_num))
    print('character assets: ', character_assets)
    print('object assets: ', obj_assets)
    print(args.sampling_points // len(obj_assets))
    pallette = plotting.hls_palette(len(assets) + 2)

    # transform matrix to standard perspective camera
    R1 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    R2 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    R3 = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    frames = sorted(glob.glob(os.path.join(args.cp_root, 'images', '*.png')))

    # Decide whether to build one shared (unified) point set across all views.
    # The multiview layout lives under scene_root: view*/ (images, exr_img) and
    # cam/view*/. Views are auto-detected from cam/view*.
    mv_base = scene_root
    view_cam_dirs = sorted(glob.glob(os.path.join(mv_base, 'cam', 'view*')))
    unified = args.unified_tracking and len(view_cam_dirs) > 1
    if args.unified_tracking and not unified:
        print('WARNING: --unified_tracking set but found {} view(s) under {}; '
              'falling back to per-view selection.'.format(len(view_cam_dirs), mv_base))
    index_cache = os.path.join(mv_base, 'tracking_index_cache.npz')

    tracking_index_list = []
    tracking_results = []
    tracking_points = None  # filled in (and shared across views) only when unified

    if unified and os.path.exists(index_cache):
        # A previous view already computed the shared set: reuse it verbatim so
        # every view tracks identical vertices and identical ground points.
        print('loading shared tracking indices from {}'.format(index_cache))
        cached = np.load(index_cache, allow_pickle=True)
        tracking_index_list = list(cached['tracking_index_list'])
        tracking_points = cached['tracking_points']
    elif unified:
        # First view: build the shared set = union, across all views, of the
        # vertices each view can see, then sample tracking_num per asset.
        view_names = [os.path.basename(d) for d in view_cam_dirs]
        print('unified tracking across {} views: {}'.format(len(view_names), view_names))
        for idx in range(len(assets)):
            asset = assets[idx]
            asset_name = asset.replace('.', '_')
            tracking_num = character_sampling_num if asset in character_assets else obj_sampling_num
            union = set()
            for view_name in view_names:
                v_exr_root = os.path.join(mv_base, view_name, 'exr_img')
                v_cam_root = os.path.join(mv_base, 'cam', view_name)
                v_frames = sorted(glob.glob(os.path.join(mv_base, view_name, 'images', '*.png')))
                best_valid = valid_candidate_indices(idx, asset_name, tracking_num,
                                                     obj_root, v_exr_root, v_cam_root,
                                                     v_frames, pallette, R1, R2, R3)
                union.update(best_valid.tolist())
            union = np.array(sorted(union), dtype=np.int64)

            if len(union) == 0:
                tracking_index = np.array([], dtype=np.int64)
            elif len(union) <= tracking_num:
                tracking_index = union
                print('WARNING: asset {}: only {} points visible across all {} views, '
                      'requested {} -- tracking all available.'.format(
                          asset, len(union), len(view_names), tracking_num))
            else:
                sample_idx = np.random.choice(len(union), tracking_num, replace=False)
                tracking_index = union[sample_idx]
            print('assets: {}, shared tracking points: {} (union pool {})'.format(
                asset, len(tracking_index), len(union)))
            tracking_index_list.append(tracking_index)

        # Ground/scene points: generate once and share across views.
        tracking_points = np.random.uniform(-4, 4, (args.sampling_scene_points, 3))
        tracking_points[:, 1] *= 0
        tracking_points *= args.scale

        np.savez(index_cache,
                 tracking_index_list=np.array(tracking_index_list, dtype=object),
                 tracking_points=tracking_points)
        print('saved shared tracking indices to {}'.format(index_cache))

    for idx in ([] if unified else range(len(assets))):
        asset = assets[idx]
        print('tracking asset: {}'.format(asset))
        asset_name = asset.replace('.', '_')
        tracking_index = []
        max_points = 0
        for i in range(0, len(frames) - 2):
            if not os.path.exists(os.path.join(obj_root, '{}_{}.obj'.format(asset_name, str(i + 2).zfill(4)))):
                continue
            initial_depth = read_tiff(os.path.join(exr_root, 'depth_{}.tiff'.format(str(i + 1).zfill(5))))
            initial_mask = cv2.imread(os.path.join(exr_root, 'segmentation_{}.png'.format(str(i + 1).zfill(5))))
            initial_rgb = cv2.imread(os.path.join(exr_root, 'rgb_{}.png'.format(str(i + 1).zfill(5))))

            h, w, _ = initial_depth.shape

            inital_points, _ = read_obj_file(os.path.join(obj_root, '{}_{}.obj'.format(asset_name, str(i + 1).zfill(4))))
            inital_K = np.loadtxt(os.path.join(cam_root, 'K_{}.txt'.format(str(i + 1).zfill(4))))
            inital_RT = np.loadtxt(os.path.join(cam_root, 'RT_{}.txt'.format(str(i + 1).zfill(4))))
            inital_RT = R3 @ R2 @ inital_RT @ R1


            uv, z = reprojection(inital_points, inital_K, inital_RT, h, w)

            uv = np.round(uv).astype(np.int32)

            asset_mask = np.logical_and(initial_mask[:, :, 2] == pallette[idx + 1, 0],
                                        initial_mask[:, :, 1] == pallette[idx + 1, 1])
            asset_mask = np.logical_and(asset_mask, initial_mask[:, :, 0] == pallette[idx + 1, 2])

            initial_depth = initial_depth

            uv_mask_0 = np.logical_and(np.logical_and(uv[:, 0] >= 2, uv[:, 0] < w - 3), np.logical_and(uv[:, 1] >= 2, uv[:, 1] < h - 3))[:, None]
            uv_idx_0 = np.where(uv_mask_0)[0]
            if len(uv_idx_0) == 0:
                continue
            uv = uv[uv_idx_0]
            uv_mask = z[uv_idx_0] <= initial_depth[uv[:, 1], uv[:, 0]]

            # to make sure points are not on the boundary
            for delta_u in range(-2, 3):
                for delta_v in range(-2, 3):
                    uv_mask = np.logical_and(uv_mask, 0.995 * z[uv_idx_0] <= initial_depth[uv[:, 1] + delta_v, uv[:, 0] + delta_u])

            uv_idx = np.where(uv_mask)[0]
            tracking_num = character_sampling_num if asset in character_assets else obj_sampling_num

            if len(uv_idx) and max_points < len(uv_idx):
                max_points = len(uv_idx)

                if len(uv_idx) <= tracking_num:
                    sample_idx = np.arange(len(uv_idx))
                else:
                    sample_idx = np.random.choice(np.arange(len(uv_idx)), tracking_num, replace=False)
                if args.farthest_sampling:
                    _, sample_idx = farthest_point_sampling(inital_points[uv_idx_0][uv_idx], args.sampling_points)

                tracking_index = uv_idx_0[uv_idx[sample_idx]]
            if (len(uv_idx) >= 0.05 * len(inital_points) and i > len(frames) // 3) or len(uv_idx) > tracking_num:
                print('assets: {}, frame: {}, tracking points: {}'.format(asset, i + 1, len(tracking_index)))
                break
        print('assets: {}, tracking points: {}'.format(asset, len(tracking_index)))
        tracking_index_list.append(tracking_index)


    tracking_results = tracking(args.cp_root, args.data_root, tracking_index_list, pallette, args.sampling_scene_points,
                                args.depth_max, args.scale, temporal_window=args.temporal_window, frame_num=args.save_num,
                                cam_root = args.cam_root, obj_root = args.obj_root, tracking_points=tracking_points)

