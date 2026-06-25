import torch
from torch import nn, Tensor, Dict



class Head2MotionComputeLosses(nn.Module):
    def __init__(self,
                 lambda_keypoints: float = 0.0,
                 lambda_theta: float = 0.0,
                 lambda_kl: float = 0.0,
                 lambda_z: float = 0.0,
                 lambda_betas: float = 0.0,
                 lambda_contacts: float = 0.0,
                 lambda_velocity: float = 0.0,
                 lambda_skating: float = 0.0,
                 kl_anneal_end_step: int = 0):
        super().__init__()
        
        self.lambda_keypoints = lambda_keypoints
        self.lambda_theta = lambda_theta
        self.lambda_kl = lambda_kl
        self.lambda_z = lambda_z
        self.lambda_betas = lambda_betas
        self.lambda_contacts = lambda_contacts
        self.lambda_velocity = lambda_velocity
        self.lambda_skating = lambda_skating
        
        self.kl_anneal_end_step = kl_anneal_end_step

        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, model_outputs, gt_motion, head_joints, motion_joints, gt_joints, batch, epoch) -> Dict[str, Tensor]:
        losses = { }
        total_loss = 0.0

        # JOINT POSITION
        if self.lambda_keypoints > 0.0:
            head_keypoints_aligned = (head_joints - head_joints[:, :, [0], :])[:, :, 1:, :] # (B, T, 21, 3)
            motion_keypoints_aligned = (motion_joints - motion_joints[:, :, [0], :])[:, :, 1:, :] # (B, T, 21, 3)
            gt_keypoints_aligned = (gt_joints - gt_joints[:, :, [0], :])[:, :, 1:, :] # (B, T, 21, 3)
            
            loss_keypoints = (self.l1_loss(head_keypoints_aligned, gt_keypoints_aligned) \
                                + self.l1_loss(motion_keypoints_aligned, gt_keypoints_aligned)) * self.lambda_keypoints
            total_loss += loss_keypoints
            losses.update({'loss_keypoints': loss_keypoints})
        
        # JOINT ROTATION
        if self.lambda_theta > 0.0:
            head_theta = model_outputs["head_theta"]
            motion_theta = model_outputs["motion_theta"]
            gt_theta = gt_motion[..., 6:132]
            # gt_theta = gt_motion[..., 9:]

            loss_theta = (self.l1_loss(head_theta, gt_theta) \
                              + self.l1_loss(motion_theta, gt_theta)) * self.lambda_theta
            total_loss += loss_theta
            losses.update({'loss_theta': loss_theta})
        
        # KL DIVERGENCE
        if self.lambda_kl > 0.0:
            # print(epoch)
            progress_ratio = epoch / self.kl_anneal_end_step
            current_beta = min(1.0, progress_ratio) 
            current_lambda = self.lambda_kl * current_beta
            
            dist_head = model_outputs['head_dist']
            dist_motion = model_outputs['motion_dist']
            
            mu_ref = torch.zeros_like(dist_head.loc)
            scale_ref = torch.ones_like(dist_head.scale)
            dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            
            # loss_kl = (torch.distributions.kl_divergence(dist_head, dist_ref).mean() \
            #              + torch.distributions.kl_divergence(dist_motion, dist_ref).mean()) * current_lambda
            loss_kl = (torch.distributions.kl_divergence(dist_head, dist_ref).mean() \
                         + torch.distributions.kl_divergence(dist_motion, dist_ref).mean()) * current_lambda
            loss_kl_align = (torch.distributions.kl_divergence(dist_head, dist_motion).mean() \
                         + torch.distributions.kl_divergence(dist_motion, dist_head).mean()) * current_lambda
            total_loss += loss_kl
            total_loss += loss_kl_align
            losses.update({
                "loss_kl": loss_kl,
                "loss_kl_align": loss_kl_align,
            })
        
        # EMBEDDING Z
        if self.lambda_z > 0.0:
            head_z = model_outputs["head_z"]
            motion_z = model_outputs["motion_z"]

            loss_z = self.l1_loss(head_z, motion_z) * self.lambda_z
            total_loss += loss_z
            losses.update({'loss_z': loss_z})     
        
        # BETAS
        if self.lambda_betas > 0.0:
            head_betas = model_outputs["head_betas"]
            motion_betas = model_outputs["motion_betas"]
            gt_betas = batch.betas
            
            loss_betas = (self.l1_loss(head_betas, gt_betas) \
                       + self.l1_loss(motion_betas, gt_betas)) * self.lambda_betas
            total_loss += loss_betas
            losses.update({'loss_betas': loss_betas})
        
        # CONTACTS
        if self.lambda_contacts > 0.0:
            head_contacts = model_outputs["head_contacts"]
            motion_contacts = model_outputs["motion_contacts"]
            gt_contacts = batch.contacts
            
            loss_contacts = (self.bce_loss(head_contacts, gt_contacts).mean() \
                          + self.bce_loss(motion_contacts, gt_contacts).mean()) * self.lambda_contacts
            total_loss += loss_contacts
            losses.update({'loss_contacts': loss_contacts})
        
        # JOINT VELOCITY
        if self.lambda_velocity > 0.0:
            head_vel = head_joints[:, 1:] - head_joints[:, :-1] 
            motion_vel = motion_joints[:, 1:] - motion_joints[:, :-1] 
            gt_vel = gt_joints[:, 1:] - gt_joints[:, :-1]
            
            loss_joint_velocity = (self.l1_loss(head_vel, gt_vel) \
                                + self.l1_loss(motion_vel, gt_vel)) * self.lambda_velocity
            total_loss += loss_joint_velocity
            losses.update({'loss_joint_velocity': loss_joint_velocity})
        
        # JOINT SKATING
        if self.lambda_skating > 0.0:
            head_vel = head_joints[:, 1:] - head_joints[:, :-1]
            head_contacts_prob = torch.sigmoid(head_contacts)
            head_vel_for_contact = head_vel[:, :, 1:, :]
            head_contacts_aligned = head_contacts_prob[:, :-1, :]
            horizontal_vel = head_vel_for_contact[..., :2]
            horizontal_speed_sq = torch.norm(horizontal_vel, dim=-1) ** 2

            motion_vel = motion_joints[:, 1:] - motion_joints[:, :-1]
            motion_contacts_prob = torch.sigmoid(motion_contacts)
            motion_vel_for_contact = motion_vel[:, :, 1:, :]
            motion_contacts_aligned = motion_contacts_prob[:, :-1, :]
            # horizontal_vel = motion_vel_for_contact[..., :2]
            # horizontal_speed_sq = torch.norm(horizontal_vel, dim=-1) ** 2
            motion_horizontal_vel = motion_vel_for_contact[..., :2]
            motion_horizontal_speed_sq = torch.norm(motion_horizontal_vel, dim=-1) ** 2

            loss_skating = ((head_contacts_aligned * horizontal_speed_sq).mean() \
                         + (motion_contacts_aligned * motion_horizontal_speed_sq).mean()) * self.lambda_skating
            total_loss += loss_skating
            losses.update({'loss_skating': loss_skating})

        losses.update({'loss_total': total_loss})
        return losses