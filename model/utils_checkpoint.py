import os
import torch
from pathlib import Path

def save_checkpoint(epoch, model, optimizer, loss, save_path):
    if (epoch+1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    p = Path(save_path)
    latest_path = p.parent / "_latest.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, latest_path)
    

def load_checkpoint(model, load_path):
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() 
        print(f"Model loaded from {load_path} for evaluation.")
        return model
    else:
        print(f"No checkpoint found at {load_path}.")
        return None
    
def load_checkpoint_resume(model, optimizer, load_path):

    if os.path.exists(load_path):
        checkpoint = torch.load(load_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        last_loss = checkpoint['loss']

        print(f"✅ Checkpoint loaded from '{load_path}'. Resuming training from epoch {start_epoch}.")
        
        return model, optimizer, start_epoch, last_loss
    else:
        print(f"🔎 No checkpoint found at '{load_path}'. Starting training from scratch.")
        
        start_epoch = 0
        last_loss = None

        return model, optimizer, start_epoch, last_loss