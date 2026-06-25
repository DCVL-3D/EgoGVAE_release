import os
os.environ['PYOPENGL_PLATFORM'] = 'egl' #'osmesa'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import cv2
import torch.nn.functional as F

from .render_openpose import render_openpose
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRenderer as Pytorch3dMeshRenderer, MeshRasterizer, SoftPhongShader,
    PointLights, TexturesVertex
)

def create_raymond_lights():
    import pyrender
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

class MeshRenderer:

    def __init__(self, cfg, faces):
        self.device = torch.device('cpu')
        self.gpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.img_res = 768
        self.faces_np = faces.astype(np.int64)

        R, T = look_at_view_transform(dist=2.5, elev=90, azim=0)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=self.img_res, blur_radius=0.0, faces_per_pixel=1
        )
        lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]])
        shader = SoftPhongShader(device=self.device, cameras=cameras, lights=lights)
        
        self.renderer = Pytorch3dMeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=shader
        )
    def visualize(self, vertices, camera_translation, images, focal_length=None, nrow=3, padding=2):
        images_np = np.transpose(images, (0,2,3,1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            fl = self.focal_length
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=False), (2,0,1))).float()
            rend_img_side = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=True), (2,0,1))).float()
            rend_imgs.append(torch.from_numpy(images[i]))
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
        rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
        return rend_imgs

    def visualize_tensorboard(self, vertices, camera_translation, images, pred_keypoints, gt_keypoints, focal_length=None, nrow=5, padding=2):
        images_np = np.transpose(images, (0,2,3,1))
        rend_imgs = []
        nrow = nrow-1 if gt_keypoints is None else nrow
        nrow = nrow-1 if pred_keypoints is None else nrow
        if pred_keypoints is not None:
            pred_keypoints = np.concatenate((pred_keypoints, np.ones_like(pred_keypoints)[:, :, [0]]), axis=-1)
            pred_keypoints = self.img_res * (pred_keypoints + 0.5)
        if gt_keypoints is not None:
            gt_keypoints[:, :, :-1] = self.img_res * (gt_keypoints[:, :, :-1] + 0.5)
        keypoint_matches = [(1, 12), (2, 8), (3, 7), (4, 6), (5, 9), (6, 10), (7, 11), (8, 14), (9, 2), (10, 1), (11, 0), (12, 3), (13, 4), (14, 5)]
        for i in range(vertices.shape[0]):
            fl = self.focal_length
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=False), (2,0,1))).float()
            rend_img_side = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=True), (2,0,1))).float()

            if pred_keypoints is not None:
                body_keypoints = pred_keypoints[i, :25]
                extra_keypoints = pred_keypoints[i, -19:]
                for pair in keypoint_matches:
                    body_keypoints[pair[0], :] = extra_keypoints[pair[1], :]
                pred_keypoints_img = render_openpose(255 * images_np[i].copy(), body_keypoints) / 255
            if gt_keypoints is not None:
                body_keypoints = gt_keypoints[i, :25]
                extra_keypoints = gt_keypoints[i, -19:]
                for pair in keypoint_matches:
                    if extra_keypoints[pair[1], -1] > 0 and body_keypoints[pair[0], -1] == 0:
                        body_keypoints[pair[0], :] = extra_keypoints[pair[1], :]
                gt_keypoints_img = render_openpose(255*images_np[i].copy(), body_keypoints) / 255
            rend_imgs.append(torch.from_numpy(images[i]))
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
            if pred_keypoints is not None:
                rend_imgs.append(torch.from_numpy(pred_keypoints_img).permute(2,0,1))
            if gt_keypoints is not None:
                rend_imgs.append(torch.from_numpy(gt_keypoints_img).permute(2,0,1))
        rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
        return rend_imgs

    def __call__(self, vertices):

        vertices_gpu = vertices.to(device=self.gpu_device, dtype=torch.float32)
        faces_gpu = torch.from_numpy(self.faces_np).to(self.gpu_device)
        
        if vertices_gpu.dim() == 2:
            vertices_gpu = vertices_gpu.unsqueeze(0)
        
        batch_size = vertices_gpu.shape[0]
        textures = TexturesVertex(verts_features=torch.ones_like(vertices_gpu) * 0.9)
        
        meshes = Meshes(
            verts=vertices_gpu, 
            faces=faces_gpu.expand(batch_size, -1, -1),
            textures=textures
        )

        rendered_images = self.renderer.to(self.gpu_device)(meshes)
        
        rgb_images = rendered_images[..., :3].permute(0, 3, 1, 2).cpu()
        
        return torch.clamp(rgb_images, 0.0, 1.0)