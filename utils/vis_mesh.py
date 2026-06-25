
import torch
import cv2
import numpy as np
import imageio

# Renderer
from utils.render_Mesh import MeshRenderer as Renderer
from utils.render_henu import MeshRenderer as Renderer_Henu


def visualize_result_png(img_path, pred_mesh, gt_mesh, smplh):
    mesh_renderer = Renderer(None, smplh.faces)
    t = 30

    head_vertices_to_vis = pred_mesh[[0], t]
    gt_vertices_to_vis = gt_mesh[[0], t]
    
    
    head_rendered_images = mesh_renderer(head_vertices_to_vis.float())
    gt_rendered_images = mesh_renderer(gt_vertices_to_vis.float())

    combined_frame = torch.cat([gt_rendered_images[0], head_rendered_images[0]], dim=2)

    frame_np = combined_frame.detach().cpu().numpy().transpose(1, 2, 0) * 255
    frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, frame_bgr)
    


def visualize_retult_mp4(video_path, pred_vertices, gt_vertices, smplh):
    
    B, T, _, _ = pred_vertices.shape
    
    mesh_renderer = Renderer_Henu(None, smplh.faces)
    
    gt_vertices[[0], : , :, 2] = gt_vertices[[0], : , :, 2] - gt_vertices[[0], : , :, 2].min()
    pred_vertices[[0], : , :, 2] = pred_vertices[[0], : , :, 2] - pred_vertices[[0], : , :, 2].min()
    
    
    temp_pred_img = mesh_renderer(gt_vertices[0, 0].float().unsqueeze(0))
    height, width = temp_pred_img.shape[2], temp_pred_img.shape[3] * 2

    fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for t in range(T):
        gt_vertices_t = gt_vertices[0, t].unsqueeze(0)
        gt_rendered_image_t = mesh_renderer(gt_vertices_t.float())
        
        pred_vertices_t = pred_vertices[0, t].unsqueeze(0)
        pred_rendered_image_t = mesh_renderer(pred_vertices_t.float())
        combined_frame = torch.cat([gt_rendered_image_t[0], pred_rendered_image_t[0]], dim=2)


        frame_np = combined_frame.cpu().numpy().transpose(1, 2, 0) * 255
        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        video_writer.write(frame_bgr)

    video_writer.release()
    print(f"Saved video to {video_path}")



def visualize_retult_gif(gif_path, pred_vertices, gt_vertices, smplh):
    
    B, T, _, _ = pred_vertices.shape
    


    # for b in range(B):
    #     mesh_renderer = Renderer_Henu(None, smplh.faces)
        
    #     gt_vertices[[b], : , :, 2] = gt_vertices[[b], : , :, 2] - gt_vertices[[b], : , :, 2].min()
    #     pred_vertices[[b], : , :, 2] = pred_vertices[[b], : , :, 2] - pred_vertices[[b], : , :, 2].min()
        
    #     temp_pred_img = mesh_renderer(gt_vertices[0, 0].float().unsqueeze(0))
    #     height, width = temp_pred_img.shape[2], temp_pred_img.shape[3] * 2

    #     fps = 20
    #     frames_for_gif = []
    #     for t in range(T):
    #         gt_vertices_t = gt_vertices[b, t].unsqueeze(0)
    #         gt_rendered_image_t = mesh_renderer(gt_vertices_t.float())
            
    #         pred_vertices_t = pred_vertices[b, t].unsqueeze(0)
    #         pred_rendered_image_t = mesh_renderer(pred_vertices_t.float())
    #         combined_frame = torch.cat([gt_rendered_image_t[0], pred_rendered_image_t[0]], dim=2)

    #         frame_np = combined_frame.cpu().numpy().transpose(1, 2, 0) * 255
    #         frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
            

    #         frames_for_gif.append(frame_np)

    #     imageio.mimsave(f'{gif_path}_{b}.gif', frames_for_gif, fps=fps)
    #     print(f"Saved GIF to {gif_path}_{b}.gif")

    mesh_renderer = Renderer_Henu(None, smplh.faces)
    
    gt_vertices[[0], : , :, 2] = gt_vertices[[0], : , :, 2] - gt_vertices[[0], : , :, 2].min()
    pred_vertices[[0], : , :, 2] = pred_vertices[[0], : , :, 2] - pred_vertices[[0], : , :, 2].min()
    
    temp_pred_img = mesh_renderer(gt_vertices[0, 0].float().unsqueeze(0))
    height, width = temp_pred_img.shape[2], temp_pred_img.shape[3] * 2

    fps = 20
    frames_for_gif = []
    for t in range(T):
        gt_vertices_t = gt_vertices[0, t].unsqueeze(0)
        gt_rendered_image_t = mesh_renderer(gt_vertices_t.float())
        
        pred_vertices_t = pred_vertices[0, t].unsqueeze(0)
        pred_rendered_image_t = mesh_renderer(pred_vertices_t.float())
        combined_frame = torch.cat([gt_rendered_image_t[0], pred_rendered_image_t[0]], dim=2)

        frame_np = combined_frame.cpu().numpy().transpose(1, 2, 0) * 255
        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        

        frames_for_gif.append(frame_np)

    imageio.mimsave(gif_path, frames_for_gif, fps=fps)
    print(f"Saved GIF to {gif_path}")

