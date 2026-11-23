import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='../results/animal')
    parser.add_argument('--partnet_root', type=str, metavar='PATH', default='partnet/')
    parser.add_argument('--gso_root', type=str, metavar='PATH', default='./data/GSO/')
    parser.add_argument('--background_hdr_path', type=str, default='./data/hdri/')
    parser.add_argument('--animal_root', type=str, default='./data/deformingthings4d')
    parser.add_argument('--animal_texture_root', type=str, default='./data/animal_textures')
    parser.add_argument('--camera_root', type=str, metavar='PATH',
                        default='./data/camera_trajectory/MannequinChallenge')
    parser.add_argument('--num_assets', type=int, default=5)
    parser.add_argument('--use_gpu', default=False, action='store_true')
    parser.add_argument('--render_engine', type=str, default='CYCLES')
    parser.add_argument('--force_num', type=int, default=5)
    parser.add_argument('--add_force', default=False, action='store_true')
    parser.add_argument('--force_step', type=int, default=3)
    parser.add_argument('--force_interval', type=int, default=120)
    parser.add_argument('--material_path', type=str, default='./data/blender_assets/animal_material.blend')
    parser.add_argument('--scene_root', type=str, default='./data/blender_assets/hdri_plane.blend')
    parser.add_argument('--add_smoke', default=False, action='store_true')
    parser.add_argument('--animal_name', type=str, metavar='PATH', default=None)
    parser.add_argument('--views', type=int, default=1)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--use_static_cam', action='store_true', help='Use static camera instead of trajectory files')
    parser.add_argument('--seed', type=int, default=None)

    # rendering settings
    parser.add_argument('--rendering',  default=False, action='store_true')


    # exr settings
    parser.add_argument('--exr',  default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--frame_idx', type=int, default=1)

    # export obj settings
    parser.add_argument('--export_obj',  default=False, action='store_true')

    # export tracking settings
    parser.add_argument('--export_tracking',  default=False, action='store_true')
    parser.add_argument('--sampling_points', type=int, default=5000)
    parser.add_argument('--sampling_scene_points', type=int, default=1000)
    parser.add_argument('--unified_tracking', default=False, action='store_true',
                        help='Track one shared point set across all views (forwarded to gen_tracking).')
    parser.add_argument('--tracking_seed', type=int, default=None,
                        help='Seed for tracking point sampling/ground points (forwarded to gen_tracking --seed). '
                             'Unset -> varies per run; set -> reproducible.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    current_path = os.path.dirname(os.path.realpath(__file__))

    if args.rendering:
        rendering_script = 'blender --background --python {}/render_animal.py \
        -- --output_dir {} --partnet_root {} --gso_root {} --background_hdr_path {} ' \
                           '--scene_root {} --animal_root {} --animal_texture_root {} --camera_root {} ' \
                           '--num_assets {} --render_engine {} --force_num {} ' \
                           '--force_step {} --force_interval {} --material_path {} --views {}'.format(
            current_path, args.output_dir, args.partnet_root, args.gso_root,
            args.background_hdr_path, args.scene_root, args.animal_root, 
            args.animal_texture_root, args.camera_root, args.num_assets,
            args.render_engine, args.force_num, args.force_step, 
            args.force_interval, args.material_path, args.views)
        if args.use_gpu:
            rendering_script += ' --use_gpu'
        if args.add_force:
            rendering_script += ' --add_force'
        if args.add_smoke:
            rendering_script += ' --add_smoke'
        if args.animal_name is not None:
            rendering_script += ' --animal_name {}'.format(args.animal_name)
        if args.max_frames is not None:
            rendering_script += ' --max_frames {}'.format(args.max_frames)
        if args.use_static_cam: 
            rendering_script += ' --use_static_cam'
        if args.seed is not None:
            rendering_script += ' --seed {}'.format(args.seed)  
        os.system(rendering_script)
    if args.export_obj:
        obj_script = 'blender --background --python {}/utils/export_obj.py \
         -- --scene_root {} --output_dir {}'.format(
            current_path, os.path.join(args.output_dir, 'scene.blend'), args.output_dir)
        os.system(obj_script)

    if args.exr:

        if args.views > 1:
            for i in range(args.views):
                view_dir = os.path.join(args.output_dir, 'view{}'.format(i))
                exr_script = 'cd {} && python -m utils.openexr_utils --data_dir {} --output_dir {} --scene_root {} --batch_size {} --frame_idx {}'.format(
                    current_path,
                    view_dir,
                    os.path.join(view_dir, 'exr_img'),
                    args.output_dir,
                    args.batch_size, args.frame_idx)
                os.system(exr_script)
                
        else:
            exr_script = 'cd {} && python -m utils.openexr_utils --data_dir {} --output_dir {} --batch_size {} --frame_idx {}'.format(
                current_path,
                os.path.join(args.output_dir, 'exr'),
                os.path.join(args.output_dir, 'exr_img'),
                args.batch_size, args.frame_idx)
            os.system(exr_script)

    if args.export_tracking:

        if args.views > 1:
            for i in range(args.views):
                view_dir = os.path.join(args.output_dir, 'view{}'.format(i))
                cam_dir = os.path.join(args.output_dir, 'cam', 'view{}'.format(i))
                tracking_script = 'cd {} && python -m utils.gen_tracking --data_root {} --cp_root {} --obj_root {} --cam_root {} --scene_root {} --sampling_points {} --sampling_scene_points {}'.format(
                    current_path, view_dir, view_dir,
                    os.path.join(args.output_dir, 'obj'),
                    cam_dir, args.output_dir,
                    args.sampling_points, args.sampling_scene_points)
                if args.unified_tracking:
                    tracking_script += ' --unified_tracking'
                if args.tracking_seed is not None:
                    tracking_script += ' --seed {}'.format(args.tracking_seed)
                os.system(tracking_script)
        else:
            tracking_script = 'cd {} && python -m utils.gen_tracking --data_root {} --cp_root {} --sampling_points {} --sampling_scene_points {}'.format(
                current_path, args.output_dir, args.output_dir, args.sampling_points, args.sampling_scene_points)
            if args.tracking_seed is not None:
                tracking_script += ' --seed {}'.format(args.tracking_seed)
            os.system(tracking_script)

    # rm redundant files
    current_path = os.path.dirname(os.path.realpath(__file__))
    output_dir_abs = os.path.abspath(args.output_dir)
    os.system('cd {} && rm -rf ./*.blend1 && rm -rf tmp'.format(output_dir_abs))

    if args.views > 1:
        for i in range(args.views):
            os.system('cd {} && rm -rf view{}/exr'.format(output_dir_abs, i))
    else:
        os.system('cd {} && rm -rf exr'.format(output_dir_abs))
    os.system('cd {}'.format(current_path))