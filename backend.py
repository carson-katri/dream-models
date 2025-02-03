from dream_textures.generator_process.actor import Actor
from dream_textures import image_utils
from dream_textures.generator_process.future import Future
import numpy as np

from dataclasses import dataclass

import site
import sys
import os
from multiprocessing import current_process

from .absolute_path import absolute_path

def _load_dependencies():
    site.addsitedir(absolute_path(".python_dependencies"))
    deps = sys.path.pop(-1)
    sys.path.insert(0, deps)
    if sys.platform == 'win32':
        # fix for ImportError: DLL load failed while importing cv2: The specified module could not be found.
        # cv2 needs python3.dll, which is stored in Blender's root directory instead of its python directory.
        python3_path = os.path.abspath(os.path.join(sys.executable, "..\\..\\..\\..\\python3.dll"))
        if os.path.exists(python3_path):
            os.add_dll_directory(os.path.dirname(python3_path))

        # fix for OSError: [WinError 126] The specified module could not be found. Error loading "...\dream_textures\.python_dependencies\torch\lib\shm.dll" or one of its dependencies.
        # Allows for shm.dll from torch==2.3.0 to access dependencies from mkl==2021.4.0
        # These DLL dependencies are not in the usual places that torch would look at due to being pip installed to a target directory.
        mkl_bin = absolute_path(".python_dependencies\\Library\\bin")
        if os.path.exists(mkl_bin):
            os.add_dll_directory(mkl_bin)
    
    if os.path.exists(absolute_path(".python_dependencies.zip")):
        sys.path.insert(1, absolute_path(".python_dependencies.zip"))

main_thread_rendering = False
is_actor_process = current_process().name == "__actor__"
if is_actor_process:
    _load_dependencies()
elif {"-b", "-f", "-a"}.intersection(sys.argv):
    main_thread_rendering = True
    import bpy
    def main_thread_rendering_finished():
        # starting without -b will allow Blender to continue running with UI after rendering is complete
        global main_thread_rendering
        main_thread_rendering = False
    bpy.app.timers.register(main_thread_rendering_finished, persistent=True)

@dataclass
class PointCloudResponse:
    vertices: np.ndarray
    colors: np.ndarray
    step: int
    steps: int

@dataclass
class MeshResponse:
    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    texture: np.ndarray

class Backend(Actor):
    def point_cloud(self, input_image: np.ndarray):
        from .spar3d.spar3d.system import SPAR3D
        from .spar3d.spar3d.utils import get_device, remove_background, foreground_crop, Remover, normalize_pc_bbox, default_cond_c2w, create_intrinsic_from_fov_rad
        
        from PIL import Image
        import torch
        from contextlib import nullcontext
        import trimesh

        future = Future()
        yield future

        # Get SPAR3D model
        model_id = "stabilityai/stable-point-aware-3d"
        
        device = get_device()

        model = SPAR3D.from_pretrained(
            model_id,
            config_name="config.yaml",
            weight_name="model.safetensors",
            low_vram_mode=False
        )
        model.to(device)
        model.eval()

        # Prepare input image
        bg_remover = Remover(device=device)
        image = remove_background(
            image_utils.np_to_pil(input_image).convert("RGBA"),
            bg_remover
        )
        image = foreground_crop(image)

        # Run model phase 1
        with torch.no_grad():
            with (
                torch.autocast(device_type=device, dtype=torch.bfloat16)
                if "cuda" in device
                else nullcontext()
            ):
                mask_cond, rgb_cond = model.prepare_image(image)
                batch_size = 1

                c2w_cond = default_cond_c2w(model.cfg.default_distance).to(model.device)
                intrinsic, intrinsic_normed_cond = create_intrinsic_from_fov_rad(
                    model.cfg.default_fovy_rad,
                    model.cfg.cond_image_size,
                    model.cfg.cond_image_size,
                )

                batch = {
                    "rgb_cond": rgb_cond,
                    "mask_cond": mask_cond,
                    "c2w_cond": c2w_cond.view(1, 1, 4, 4).repeat(batch_size, 1, 1, 1),
                    "intrinsic_cond": intrinsic.to(model.device)
                    .view(1, 1, 3, 3)
                    .repeat(batch_size, 1, 1, 1),
                    "intrinsic_normed_cond": intrinsic_normed_cond.to(model.device)
                    .view(1, 1, 3, 3)
                    .repeat(batch_size, 1, 1, 1),
                }
                batch["rgb_cond"] = model.image_processor(
                    batch["rgb_cond"], model.cfg.cond_image_size
                )
                batch["mask_cond"] = model.image_processor(
                    batch["mask_cond"], model.cfg.cond_image_size
                )

                batch_size = batch["rgb_cond"].shape[0]

                cond_tokens = model.forward_pdiff_cond(batch)
                sample_iter = model.sampler.sample_batch_progressive(
                    batch_size, cond_tokens, device=model.device
                )
                step_index = 0
                for x in sample_iter:
                    samples = x["xstart"]
                    
                    denoised_pc = samples.permute(0, 2, 1).float()  # [B, C, N] -> [B, N, C]
                    denoised_pc = normalize_pc_bbox(denoised_pc)

                    # predict the full 3D conditioned on the denoised point cloud
                    batch["pc_cond"] = denoised_pc
            
                    for i in range(batch_size):
                        vertices = batch["pc_cond"][i, :, :3].cpu().numpy()
                        colors = batch["pc_cond"][i, :, 3:6].cpu().numpy()
                        colors_rgba = np.concatenate((
                            colors,
                            np.ones((*colors.shape[:-1], 1), dtype=np.float32)
                        ), axis=1).astype(np.float32)
                        future.add_response(PointCloudResponse(vertices, colors_rgba, step_index, model.sampler.diffusion.num_timesteps))
                    
                    step_index += 1
                
                future.set_done()
    
    def mesh(self, input_image: np.ndarray, input_point_cloud: np.ndarray, input_point_cloud_colors: np.ndarray):
        from .spar3d.spar3d.system import SPAR3D
        from .spar3d.spar3d.utils import get_device, remove_background, foreground_crop, Remover, normalize_pc_bbox, default_cond_c2w, create_intrinsic_from_fov_rad
        from .spar3d.spar3d.models.utils import normalize, convert_data, float32_to_uint8_np, dilate_fill
        
        from PIL import Image
        import torch
        import torch.nn.functional as F
        from contextlib import nullcontext
        import trimesh

        future = Future()
        yield future

        remesh = "none"
        vertex_count = -1
        bake_resolution = 1024

        # Get SPAR3D model
        model_id = "stabilityai/stable-point-aware-3d"
        
        device = get_device()

        model = SPAR3D.from_pretrained(
            model_id,
            config_name="config.yaml",
            weight_name="model.safetensors",
            low_vram_mode=False
        )
        model.to(device)
        model.eval()

        # Prepare input image
        bg_remover = Remover(device=device)
        image = remove_background(
            image_utils.np_to_pil(input_image).convert("RGBA"),
            bg_remover
        )
        image = foreground_crop(image)

        # run model phase 2
        with torch.no_grad():
            with (
                torch.autocast(device_type=device, enabled=False)
                if "cuda" in device
                else nullcontext()
            ):
                mask_cond, rgb_cond = model.prepare_image(image)
                batch_size = 1

                c2w_cond = default_cond_c2w(model.cfg.default_distance).to(model.device)
                intrinsic, intrinsic_normed_cond = create_intrinsic_from_fov_rad(
                    model.cfg.default_fovy_rad,
                    model.cfg.cond_image_size,
                    model.cfg.cond_image_size,
                )

                batch = {
                    "rgb_cond": rgb_cond,
                    "mask_cond": mask_cond,
                    "c2w_cond": c2w_cond.view(1, 1, 4, 4).repeat(batch_size, 1, 1, 1),
                    "intrinsic_cond": intrinsic.to(model.device)
                    .view(1, 1, 3, 3)
                    .repeat(batch_size, 1, 1, 1),
                    "intrinsic_normed_cond": intrinsic_normed_cond.to(model.device)
                    .view(1, 1, 3, 3)
                    .repeat(batch_size, 1, 1, 1),
                }
                batch["rgb_cond"] = model.image_processor(
                    batch["rgb_cond"], model.cfg.cond_image_size
                ).unsqueeze(0)
                batch["mask_cond"] = model.image_processor(
                    batch["mask_cond"], model.cfg.cond_image_size
                ).unsqueeze(0)

                batch_size = batch["rgb_cond"].shape[0]

                batch["pc_cond"] = torch.tensor(np.concatenate([input_point_cloud, input_point_cloud_colors[:, :3]], axis=1)).unsqueeze(0).float().to(device)

                scene_codes, non_postprocessed_codes = model.get_scene_codes(batch)

                # Create a rotation matrix for the final output domain
                rotation = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
                rotation2 = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
                output_rotation = rotation2 @ rotation

                global_dict = {}
                if model.is_low_vram:
                    model._unload_pdiff_modules()
                    model._unload_main_modules()
                    model._load_estimator_modules()

                if model.image_estimator is not None:
                    global_dict.update(
                        model.image_estimator(
                            torch.cat([batch["rgb_cond"], batch["mask_cond"]], dim=-1)
                        )
                    )

                meshes = model.triplane_to_meshes(scene_codes)

                rets = []
                for i, mesh in enumerate(meshes):
                    # Check for empty mesh
                    if mesh.v_pos.shape[0] == 0:
                        rets.append(trimesh.Trimesh())
                        continue

                    if remesh == "triangle":
                        mesh = mesh.triangle_remesh(triangle_vertex_count=vertex_count)
                    elif remesh == "quad":
                        mesh = mesh.quad_remesh(quad_vertex_count=vertex_count)
                    else:
                        if vertex_count > 0:
                            print(
                                "Warning: vertex_count is ignored when remesh is none"
                            )

                    if remesh != "none":
                        print(
                            f"After {remesh} remesh the mesh has {mesh.v_pos.shape[0]} verts and {mesh.t_pos_idx.shape[0]} faces",
                        )
                        mesh.unwrap_uv()

                    # Build textures
                    rast = model.baker.rasterize(
                        mesh.v_tex, mesh.t_pos_idx, bake_resolution
                    )
                    bake_mask = model.baker.get_mask(rast)

                    pos_bake = model.baker.interpolate(
                        mesh.v_pos,
                        rast,
                        mesh.t_pos_idx,
                    )
                    gb_pos = pos_bake[bake_mask]

                    tri_query = model.query_triplane(gb_pos, scene_codes[i])[0]
                    decoded = model.decoder(
                        tri_query, exclude=["density", "vertex_offset"]
                    )

                    nrm = model.baker.interpolate(
                        mesh.v_nrm,
                        rast,
                        mesh.t_pos_idx,
                    )
                    gb_nrm = F.normalize(nrm[bake_mask], dim=-1)
                    decoded["normal"] = gb_nrm

                    # Check if any keys in global_dict start with decoded_
                    for k, v in global_dict.items():
                        if k.startswith("decoder_"):
                            decoded[k.replace("decoder_", "")] = v[i]

                    mat_out = {
                        "albedo": decoded["features"],
                        "roughness": decoded["roughness"],
                        "metallic": decoded["metallic"],
                        "normal": normalize(decoded["perturb_normal"]),
                        "bump": None,
                    }

                    for k, v in mat_out.items():
                        if v is None:
                            continue
                        if v.shape[0] == 1:
                            # Skip and directly add a single value
                            mat_out[k] = v[0]
                        else:
                            f = torch.zeros(
                                bake_resolution,
                                bake_resolution,
                                v.shape[-1],
                                dtype=v.dtype,
                                device=v.device,
                            )
                            if v.shape == f.shape:
                                continue
                            if k == "normal":
                                # Use un-normalized tangents here so that larger smaller tris
                                # Don't effect the tangents that much
                                tng = model.baker.interpolate(
                                    mesh.v_tng,
                                    rast,
                                    mesh.t_pos_idx,
                                )
                                gb_tng = tng[bake_mask]
                                gb_tng = F.normalize(gb_tng, dim=-1)
                                gb_btng = F.normalize(
                                    torch.cross(gb_nrm, gb_tng, dim=-1), dim=-1
                                )
                                normal = F.normalize(mat_out["normal"], dim=-1)

                                # Create tangent space matrix and transform normal
                                tangent_matrix = torch.stack(
                                    [gb_tng, gb_btng, gb_nrm], dim=-1
                                )
                                normal_tangent = torch.bmm(
                                    tangent_matrix.transpose(1, 2), normal.unsqueeze(-1)
                                ).squeeze(-1)

                                # Convert from [-1,1] to [0,1] range for storage
                                normal_tangent = (normal_tangent * 0.5 + 0.5).clamp(
                                    0, 1
                                )

                                f[bake_mask] = normal_tangent.view(-1, 3)
                                mat_out["bump"] = f
                            else:
                                f[bake_mask] = v.view(-1, v.shape[-1])
                                mat_out[k] = f

                    def uv_padding(arr):
                        if arr.ndim == 1:
                            return arr
                        return (
                            dilate_fill(
                                arr.permute(2, 0, 1)[None, ...].contiguous(),
                                bake_mask.unsqueeze(0).unsqueeze(0),
                                iterations=bake_resolution // 150,
                            )
                            .squeeze(0)
                            .permute(1, 2, 0)
                            .contiguous()
                        )

                    verts_np = convert_data(mesh.v_pos)
                    faces = convert_data(mesh.t_pos_idx)
                    uvs = convert_data(mesh.v_tex)

                    basecolor_tex = Image.fromarray(
                        float32_to_uint8_np(convert_data(uv_padding(mat_out["albedo"])))
                    ).convert("RGB")
                    basecolor_tex.format = "JPEG"

                    metallic = mat_out["metallic"].squeeze().cpu().item()
                    roughness = mat_out["roughness"].squeeze().cpu().item()

                    if "bump" in mat_out and mat_out["bump"] is not None:
                        bump_np = convert_data(uv_padding(mat_out["bump"]))
                        bump_up = np.ones_like(bump_np)
                        bump_up[..., :2] = 0.5
                        bump_up[..., 2:] = 1
                        bump_tex = Image.fromarray(
                            float32_to_uint8_np(
                                bump_np,
                                dither=True,
                                # Do not dither if something is perfectly flat
                                dither_mask=np.all(
                                    bump_np == bump_up, axis=-1, keepdims=True
                                ).astype(np.float32),
                            )
                        ).convert("RGB")
                        bump_tex.format = (
                            "JPEG"  # PNG would be better but the assets are larger
                        )
                    else:
                        bump_tex = None
                    
                    future.add_response(MeshResponse(
                        np.array(verts_np),
                        np.array(faces),
                        np.array(uvs),
                        image_utils.pil_to_np(basecolor_tex)
                    ))
                    future.set_done()

                    # material = trimesh.visual.material.PBRMaterial(
                    #     baseColorTexture=basecolor_tex,
                    #     roughnessFactor=roughness,
                    #     metallicFactor=metallic,
                    #     normalTexture=bump_tex,
                    # )

                    # tmesh = trimesh.Trimesh(
                    #     vertices=verts_np,
                    #     faces=faces,
                    #     visual=trimesh.visual.texture.TextureVisuals(
                    #         uv=uvs, material=material
                    #     ),
                    # )
                    # tmesh.apply_transform(output_rotation)

                    # tmesh.invert()

                    # rets.append(tmesh)
        
        # print(global_dict)
        # print(rets)

        # mesh: trimesh.Trimesh = rets[0]
        # return np.array(mesh.vertices), np.array(mesh.faces)