import os
import torch
import numpy as np
import cv2
from torchvision.utils import make_grid
import trimesh

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, PerspectiveCameras,
    RasterizationSettings, MeshRenderer as Pytorch3dMeshRenderer, MeshRasterizer,
    SoftPhongShader, PointLights, TexturesVertex,
    look_at_rotation
)
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer.camera_conversions import _cameras_from_opencv_projection


def checkerboard_geometry(
    length=12.0, color0=[0.8, 0.9, 0.9], color1=[0.6, 0.7, 0.7],
    tile_width=0.5, alpha=1.0, up="z", c1=0.0, c2=0.0
):
    assert up == "y" or up == "z"
    color0 = np.array(color0 + [alpha])
    color1 = np.array(color1 + [alpha])
    radius = length / 2.0
    num_rows = num_cols = max(2, int(length / tile_width))
    vertices, vert_colors, faces, face_colors = [], [], [], []
    for i in range(num_rows):
        for j in range(num_cols):
            u0, v0 = j * tile_width - radius, i * tile_width - radius
            us = np.array([u0, u0, u0 + tile_width, u0 + tile_width])
            vs = np.array([v0, v0 + tile_width, v0 + tile_width, v0])
            zs = np.zeros(4)
            if up == "y":
                cur_verts = np.stack([us, zs, vs], axis=-1)
                cur_verts[:, 0] += c1
                cur_verts[:, 2] += c2
            else:
                cur_verts = np.stack([us, vs, zs], axis=-1)
                cur_verts[:, 0] += c1
                cur_verts[:, 1] += c2

            cur_faces = np.array(
                [[0, 1, 3], [1, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64
            )
            cur_faces += 4 * (i * num_cols + j)
            use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
            cur_color = color0 if use_color0 else color1
            cur_colors = np.array([cur_color, cur_color, cur_color, cur_color])

            vertices.append(cur_verts)
            faces.append(cur_faces)
            vert_colors.append(cur_colors)
            face_colors.append(cur_colors)

    vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    vert_colors = np.concatenate(vert_colors, axis=0).astype(np.float32)
    faces = np.concatenate(faces, axis=0).astype(np.int64)
    face_colors = np.concatenate(face_colors, axis=0).astype(np.float32)

    return vertices, faces, vert_colors, face_colors

def create_meshes(verts, faces, colors):
    textures = TexturesVertex(verts_features=colors)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)
    return join_meshes_as_scene(meshes)

def prep_shared_geometry(verts, faces, colors):
    B, V, _ = verts.shape
    F, _ = faces.shape
    colors = colors.unsqueeze(1).expand(B, V, -1)[..., :3]
    faces = faces.unsqueeze(0).expand(B, F, -1)
    return verts, faces, colors


class MeshRenderer:
    def __init__(self, cfg, faces):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.img_res = 758
        self.faces_np = faces.astype(np.int64)
        
        # --- Z-up -> Y-up 변환 행렬 ---
        # X축을 중심으로 -90도 회전
        self.zup_to_yup_rotation = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0]
        ], device=self.device).float()
        
        # --- 수정된 부분 1: 바닥면 생성 시 up="y"로 설정 ---
        # 이제 바닥면은 XZ 평면(Y=0)에 놓입니다.
        self.ground_geometry = self.set_ground(length=20.0, center_x=0, center_z=0, up="y")

        # --- 고정된 카메라 시점 사용 ---
        # 카메라 시점은 그대로 유지하며, 이제 바닥면(Y=0)을 올바르게 내려다봅니다.
        R, T = look_at_view_transform(dist=5, elev=20, azim=0)
        # R, T = look_at_view_transform(dist=5, elev=1, azim=0)
        
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        
        raster_settings = RasterizationSettings(
            image_size=self.img_res, blur_radius=0.0, faces_per_pixel=1
        )
        lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]])
        shader = SoftPhongShader(device=self.device, cameras=self.cameras, lights=lights)
        
        self.renderer = Pytorch3dMeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
            shader=shader
        )
        
    def set_ground(self, length, center_x, center_z, up="y"):
        v, f, vc, _ = checkerboard_geometry(length=length, c1=center_x, c2=center_z, up=up)
        
        v, f, vc = map(torch.from_numpy, [v, f, vc])
        v, f, vc = v.to(self.device), f.to(self.device), vc[..., :3].to(self.device)
        return [v, f, vc]

    def __call__(self, vertices: torch.Tensor, return_separate=False):
        B, V, _ = vertices.shape
        
        # --- 수정된 부분: 인체 메쉬 버텍스에 Z-up -> Y-up 변환 적용 ---
        # 원래 데이터가 Z-up 컨벤션이라고 가정하고, Y-up으로 변환합니다.
        vertices_yup = vertices @ self.zup_to_yup_rotation.T

        # 인간 메쉬 생성
        human_textures = TexturesVertex(verts_features=torch.ones_like(vertices_yup) * 0.9)
        human_meshes = Meshes(
            verts=vertices_yup.to(self.device).float(), 
            faces=torch.from_numpy(self.faces_np).expand(B, -1, -1).to(self.device),
            textures=human_textures
        )

        # 바닥 메쉬 생성
        ground_verts, ground_faces, ground_colors = self.ground_geometry
        ground_colors_expanded = ground_colors.unsqueeze(0).expand(B, -1, -1)
        ground_textures = TexturesVertex(verts_features=ground_colors_expanded)
        
        ground_meshes = Meshes(
            verts=ground_verts.unsqueeze(0).expand(B, -1, -1),
            faces=ground_faces.unsqueeze(0).expand(B, -1, -1),
            textures=ground_textures
        )
        
        # 인간 메쉬와 바닥 메쉬를 하나의 씬으로 병합
        combined_meshes = join_meshes_as_scene([human_meshes, ground_meshes])

        # 렌더링
        rendered_images = self.renderer(combined_meshes, cameras=self.cameras)
        
        # 출력 형식 맞추기
        rgb_images = rendered_images[..., :3].permute(0, 3, 1, 2).cpu().reshape(B, 3, self.img_res, self.img_res)
        
        return torch.clamp(rgb_images, 0.0, 1.0)