"""
Attentive Probing for RE10K Relative Pose Estimation

This script implements attentive probing experiment for 6DoF relative camera pose prediction
on RealEstate10k dataset, following the training paradigm from KITTI/ScanNet depth estimation probing.

Architecture:
- LearnedQueries (num_channels=256)
- CrossAttention (qkv_size=256, num_heads=8)  
- Linear (output_size=12) for pose prediction

Training Setup (following KITTI/ScanNet):
- Learning rate: 3×10⁻³
- Batch size: 128
- Epochs: 30
- Encoder: Frozen (only readout head trained)
"""

import argparse
import math
import os
import time
import warnings
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
from timm.models.layers import trunc_normal_
from torch import distributed
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoModel

# Import custom models and layers
import model_factory
try:
    from dataloader.re10k_dataloader_dali import get_re10k_dataloader_dali
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    # Fallback to PyTorch dataloader
    try:
        from dataloader.re10k_dataloader import get_re10k_dataloader_dali
        print("Warning: DALI not available, falling back to PyTorch dataloader")
    except ImportError:
        print("Error: No dataloader available")

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Attentive Probing - RE10K Relative Pose Estimation")
    
    # Data

    parser.add_argument("--re10k_dir", default="/video_vit/feilong/3dmulti/datasets/Re10K", 
                       help="Path to RealEstate10k frames directory")
    parser.add_argument("--re10k_annotation_dir", default="/video_vit/feilong/3dmulti/datasets/Re10K-annotations/RealEstate10K",
                       help="Path to RealEstate10k annotations")
    parser.add_argument("--split", default="train", choices=["train", "test"])

    # Model
    parser.add_argument("--model_family", default="llava_vit_sampling")
    parser.add_argument("--model_name", default="llava_vit_base_ln")
    parser.add_argument("--model_weight", default="NULL")
    parser.add_argument("--num_frames", type=int, default=16,
                       help="Number of frames (16 frames for feature extraction, using first and last for pose)")
    parser.add_argument("--probe_num_frames", type=int, default=4,
                       help="Number of probe frames (S=4: 1 reference + 3 auxiliary frames)")
    parser.add_argument("--probe_min_gap", type=int, default=5,
                       help="Minimum frame gap between reference and auxiliary frames in probe sampling")
    parser.add_argument("--max_context_window", type=int, default=16,
                       help="Maximum context window (number of frames) supported by VidFM per chunk")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--embedding_size", type=int, default=768)
    parser.add_argument("--pose_hidden_dim", type=int, default=256,
                       help="Hidden dimension for pose estimation head")
    
    # Training (following KITTI/ScanNet probing experiment setup)
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for training (128 for RE10K, similar to KITTI)")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of DataLoader workers per GPU (increased from 4 to 8)")
    parser.add_argument("--default_epoch", type=int, default=30,
                       help="Number of training epochs (30 for convergence)")
    parser.add_argument("--default_weight_decay", type=float, default=0.01)
    parser.add_argument("--default_min_lr", type=float, default=1e-6)
    parser.add_argument("--default_lr_list", type=float, nargs="+", default=[3e-3],
                       help="Learning rate (3e-3 following KITTI/ScanNet setup)")
    parser.add_argument("--clip_grad", type=float, default=5.0)
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=1)
    
    # Dataloader options
    parser.add_argument("--dali_num_threads", type=int, default=8,
                       help="Number of threads per DALI pipeline (default: 8, increase to 16 if CPU-bound)")
    parser.add_argument("--dali_py_num_workers", type=int, default=4,
                       help="Number of Python workers for DALI external source (default: 4, conservative for multi-GPU; increase to 8 if memory available)")
    parser.add_argument("--image_cache_size", type=int, default=100,
                       help="Number of images to cache per worker (default 100, conservative for multi-GPU; increase to 250 if memory available)")
    parser.add_argument("--prefetch_queue_depth", type=int, default=1,
                       help="Number of batches to prefetch in DALI pipeline (default 1, conservative for multi-GPU; increase to 2 if memory available)")
    parser.add_argument("--py_start_method", type=str, default="spawn",
                       choices=["spawn", "fork", "forkserver"],
                       help="Process start method for DALI workers (default: spawn, safest option)")
    
    # Misc
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_report", default="fewshot_video_report/PoseEstimation")
    
    # Distributed
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    
    return parser.parse_args()


class RelativePoseHead(nn.Module):
    """
    Attentive probing head for relative pose estimation with SVD orthogonalization.
    
    Architecture:
    - LearnedQueries (num_channels=256)
    - CrossAttention (qkv_size=256, num_heads=8)
    - Linear (output_size=12)
    - SVD orthogonalization applied to rotation matrix during training to ensure SO(3) constraint
    """
    def __init__(self, hidden_dim: int, pose_hidden_dim: int = 256, init_scale: float = 1e-3):
        super().__init__()
        
        # Learned query for pose estimation (single query)
        self.learned_query = nn.Parameter(torch.zeros(1, 1, pose_hidden_dim))
        trunc_normal_(self.learned_query, std=0.02)
        
        # Project encoder features to pose_hidden_dim
        self.feature_proj = nn.Linear(hidden_dim, pose_hidden_dim)
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=pose_hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Layer norm
        self.norm1 = nn.LayerNorm(pose_hidden_dim)
        self.norm2 = nn.LayerNorm(pose_hidden_dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(pose_hidden_dim, pose_hidden_dim * 4),
            nn.GELU(),
            nn.Linear(pose_hidden_dim * 4, pose_hidden_dim)
        )
        
        # Final projection to 12-dim pose (flattened 3x3 rotation matrix [9 elements] + 3x1 translation [3 elements])
        self.pose_head = nn.Linear(pose_hidden_dim, 12)
        
        # Initialize weights
        self.apply(self._init_weights)
        self.pose_head.weight.data.mul_(init_scale)
        self.pose_head.bias.data.mul_(init_scale)
    
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def svd_orthogonalize(self, m: torch.Tensor) -> torch.Tensor:
        """
        Convert 9D representation to SO(3) using SVD orthogonalization.
        This ensures the rotation matrix satisfies SO(3) constraints during training.

        Args:
          m: [BATCH, 3, 3] or [BATCH, 9] rotation matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        # Store original dtype for conversion back
        original_dtype = m.dtype
        
        # Convert to float32 if needed (SVD doesn't support bfloat16)
        if m.dtype == torch.bfloat16:
            m = m.float()
        
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        
        # Normalize rows first
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        
        # SVD decomposition
        u, s, v = torch.svd(m_transpose)
        
        # Compute determinant to check for reflections
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)).float())
        
        # Ensure proper rotation (det = 1, not reflection with det = -1)
        # If det is negative, flip the last column of v
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        
        # Convert back to original dtype
        if original_dtype == torch.bfloat16:
            r = r.to(original_dtype)
        
        return r
    
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: [B, seq_len, hidden_dim] - Encoder features from first and last frame
        
        Returns:
            pose: [B, 12] - Predicted relative pose (9 rotation + 3 translation)
                           Rotation is SVD-orthogonalized to SO(3) during training
        """
        B = feats.shape[0]
        
        # Project features to pose_hidden_dim
        feats = self.feature_proj(feats)  # [B, seq_len, pose_hidden_dim]
        
        # Expand learned query for batch
        query = self.learned_query.expand(B, -1, -1)  # [B, 1, pose_hidden_dim]
        
        # Cross-attention: query attends to encoder features
        attn_out, _ = self.cross_attention(
            query=query,
            key=feats,
            value=feats
        )  # [B, 1, pose_hidden_dim]
        
        # Residual connection and norm
        query = self.norm1(query + attn_out)
        
        # MLP with residual
        mlp_out = self.mlp(query)
        query = self.norm2(query + mlp_out)  # [B, 1, pose_hidden_dim]
        
        # Project to pose
        pose = self.pose_head(query.squeeze(1))  # [B, 12]
        
        # During training, apply SVD orthogonalization to ensure SO(3) constraint
        if self.training:
            # Split into rotation (9D) and translation (3D)
            out_r = pose[:, :9]  # [B, 9]
            out_t = pose[:, 9:]  # [B, 3]
            
            # Apply SVD orthogonalization to rotation
            out_r_mat = self.svd_orthogonalize(out_r)  # [B, 3, 3]
            
            # Flatten back and concatenate
            pose = torch.cat([out_r_mat.view(B, -1), out_t], dim=-1)  # [B, 12]
        
        return pose


def procrustes_rotation(R: torch.Tensor) -> torch.Tensor:
    """
    Apply Procrustes method to project estimated rotation matrix to SO(3).
    
    Args:
        R: [B, 3, 3] - Estimated rotation matrix (may not be in SO(3))
    
    Returns:
        R_proj: [B, 3, 3] - Projected rotation matrix in SO(3)
    """
    # SVD: R = U @ S @ V^T
    U, S, Vt = torch.linalg.svd(R)
    
    # Closest rotation matrix: R_proj = U @ V^T
    R_proj = torch.bmm(U, Vt)
    
    # Ensure det(R_proj) = 1 (proper rotation, not reflection)
    det = torch.linalg.det(R_proj)
    # If det is -1, flip the last column of U
    flip = (det < 0).float()
    U_corrected = U.clone()
    U_corrected[:, :, -1] = U_corrected[:, :, -1] * (1 - 2 * flip.unsqueeze(-1))
    R_proj = torch.bmm(U_corrected, Vt)
    
    return R_proj


def pose_loss(pred_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
    """
    Compute pose loss (L2 loss on rotation and translation vectors).
    
    Args:
        pred_pose: [B, 12] - Predicted pose (9 rotation + 3 translation)
        target_pose: [B, 12] - Ground truth pose
    
    Returns:
        loss: Scalar loss value
    """
    # L2 loss on the full pose vector (as specified in the paper)
    loss = F.mse_loss(pred_pose, target_pose)
    
    return loss


def compute_epe_metric(pred_R: torch.Tensor, pred_t: torch.Tensor, 
                       target_R: torch.Tensor, target_t: torch.Tensor) -> torch.Tensor:
    """
    Compute End-Point-Error (EPE) metric for pose estimation.
    
    This metric measures the mean distance of 8 auxiliary points forming a virtual cube
    in front of the camera, transformed with ground-truth and estimated poses.
    
    Args:
        pred_R: [B, 3, 3] - Predicted rotation matrix
        pred_t: [B, 3] - Predicted translation
        target_R: [B, 3, 3] - Ground truth rotation matrix
        target_t: [B, 3] - Ground truth translation
    
    Returns:
        epe: [B] - End-point-error for each sample
    """
    batch_size = pred_R.shape[0]
    device = pred_R.device
    
    # Define 8 points forming a unit cube in front of the camera
    # Cube corners at distance 1.0 from origin with size 0.5
    cube_points = torch.tensor([
        [0.5, 0.5, 1.0],
        [0.5, -0.5, 1.0],
        [-0.5, 0.5, 1.0],
        [-0.5, -0.5, 1.0],
        [0.5, 0.5, 1.5],
        [0.5, -0.5, 1.5],
        [-0.5, 0.5, 1.5],
        [-0.5, -0.5, 1.5],
    ], dtype=pred_R.dtype, device=device)  # [8, 3]
    
    # Expand for batch: [B, 8, 3]
    cube_points = cube_points.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Transform points with predicted pose: P_pred = R @ X + t
    # pred_R: [B, 3, 3], cube_points: [B, 8, 3] -> [B, 3, 8]
    points_pred = torch.bmm(pred_R, cube_points.transpose(1, 2))  # [B, 3, 8]
    points_pred = points_pred.transpose(1, 2) + pred_t.unsqueeze(1)  # [B, 8, 3]
    
    # Transform points with ground truth pose
    points_target = torch.bmm(target_R, cube_points.transpose(1, 2))  # [B, 3, 8]
    points_target = points_target.transpose(1, 2) + target_t.unsqueeze(1)  # [B, 8, 3]
    
    # Compute EPE: mean distance across all 8 points
    epe = torch.norm(points_pred - points_target, dim=2).mean(dim=1)  # [B]
    
    return epe


def rotation_error(pred_R: torch.Tensor, target_R: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation error in degrees.
    
    Args:
        pred_R: [B, 9] - Predicted rotation matrix (flattened)
        target_R: [B, 9] - Target rotation matrix (flattened)
    
    Returns:
        error: [B] - Rotation error in degrees
    """
    # Reshape to [B, 3, 3]
    pred_R = pred_R.view(-1, 3, 3)
    target_R = target_R.view(-1, 3, 3)
    
    # Compute relative rotation error matrix: R_rel = pred_R @ target_R^T
    # This gives the rotation difference between predicted and target
    R_rel = torch.bmm(pred_R, target_R.transpose(1, 2))
    
    # Rotation angle from trace: theta = arccos((trace(R) - 1) / 2)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    
    # Convert to degrees
    error = theta * 180.0 / math.pi
    
    return error


def translation_error(pred_t: torch.Tensor, target_t: torch.Tensor) -> torch.Tensor:
    """
    Compute translation error (L2 distance).
    
    Args:
        pred_t: [B, 3] - Predicted translation
        target_t: [B, 3] - Target translation
    
    Returns:
        error: [B] - Translation error
    """
    error = torch.norm(pred_t - target_t, dim=1)
    return error


def compute_frame_to_chunk_mapping(num_frames: int, max_context_window: int) -> tuple:
    """
    Compute frame-to-chunk index mapping function π(t).
    
    Each chunk includes the global reference frame (frame 0) at the beginning,
    followed by consecutive frames from the video.
    
    Args:
        num_frames: Total number of frames in the video
        max_context_window: Maximum number of frames per chunk (including reference frame)
    
    Returns:
        frame_to_chunk: List of (chunk_id, local_idx) for each frame
                       frame_to_chunk[t] = (chunk_id, local_idx) where:
                       - chunk_id: which chunk the frame belongs to
                       - local_idx: position within that chunk
        num_chunks: Total number of chunks
        chunk_frame_lists: List of frame indices for each chunk (including reference frame)
    
    Example:
        num_frames=20, max_context_window=8
        Chunk 0: [frame_0, frame_1, frame_2, ..., frame_7]   (8 frames: ref + 7 frames)
        Chunk 1: [frame_0, frame_8, frame_9, ..., frame_14]  (8 frames: ref + 7 frames)  
        Chunk 2: [frame_0, frame_15, frame_16, ..., frame_19] (6 frames: ref + 5 frames)
        
        π(0) = (0, 0), (1, 0), (2, 0)  # Frame 0 appears in all chunks at position 0
        π(1) = (0, 1)                   # Frame 1 appears in chunk 0 at position 1
        π(8) = (1, 1)                   # Frame 8 appears in chunk 1 at position 1
    """
    if num_frames <= 1:
        # Special case: only one frame
        return [(0, 0)], 1, [[0]]
    
    # Each chunk has: 1 reference frame + (max_context_window - 1) content frames
    # Frame 0 is the global reference frame
    content_frames_per_chunk = max_context_window - 1  # Reserve 1 slot for reference frame
    
    # Calculate number of chunks needed to cover all frames (excluding frame 0 which is reference)
    num_content_frames = num_frames - 1  # Frames 1, 2, ..., num_frames-1
    num_chunks = (num_content_frames + content_frames_per_chunk - 1) // content_frames_per_chunk
    
    # Initialize mapping
    frame_to_chunk = [[] for _ in range(num_frames)]  # Each frame can appear in multiple chunks
    chunk_frame_lists = []
    
    # Build chunks
    for chunk_id in range(num_chunks):
        # Each chunk starts with the reference frame (frame 0)
        chunk_frames = [0]
        
        # Add content frames for this chunk
        start_frame = 1 + chunk_id * content_frames_per_chunk
        end_frame = min(start_frame + content_frames_per_chunk, num_frames)
        
        for frame_idx in range(start_frame, end_frame):
            chunk_frames.append(frame_idx)
        
        # Record chunk structure
        chunk_frame_lists.append(chunk_frames)
        
        # Update frame-to-chunk mapping
        for local_idx, frame_idx in enumerate(chunk_frames):
            frame_to_chunk[frame_idx].append((chunk_id, local_idx))
    
    return frame_to_chunk, num_chunks, chunk_frame_lists


def get_feature(
    args: argparse.Namespace,
    images: torch.Tensor,
    model: nn.Module,
    probe_frame_indices: torch.Tensor = None,
) -> torch.Tensor:
    """
    Extract features from images with chunking support.
    
    For videos longer than max_context_window, the video is split into chunks.
    Each chunk includes the first frame (reference frame) followed by consecutive frames.
    Features are extracted per chunk and then mapped back to original frame positions.
    
    Args:
        args: Arguments
        images: [B, T, 3, H, W] - T frames from video (already selected/sampled)
        model: Encoder model
        probe_frame_indices: [B, T] - Optional metadata tracking original video frame indices.
                            This is for bookkeeping only and not used for indexing.
                            The images tensor already contains the selected frames.
    
    Returns:
        features: [B, seq_len, hidden_dim] - Features for all T frames in images
                  For image models: seq_len = T*num_patches
                  For video models: seq_len depends on model's temporal pooling
    """
    B, T, C, H, W = images.shape
    max_context = args.max_context_window
    
    # If video fits in context window, process directly without chunking
    if T <= max_context:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                if args.model_family in ["clip", "siglip", "siglip2", "dinov2", "dinov3", "metaclip", "llava_vit_si", "aimv2"]:
                    # For image models, process frames individually
                    all_frames = images.view(B * T, C, H, W)
                    hidden_states = model(all_frames)
                    if isinstance(hidden_states, dict) and "visible_embeddings" in hidden_states:
                        hidden_states = hidden_states["visible_embeddings"]
                    # Reshape back to [B, T*seq_len, hidden_dim]
                    hidden_states = hidden_states.view(B, -1, hidden_states.size(-1))
                elif args.model_family == "llava_vit_sampling":
                    # For video models, use native video input format [B, C, T, H, W]
                    videos = images.transpose(1, 2)  # [B, C, T, H, W]
                    hidden_states = model(videos, None)
                    if hasattr(hidden_states, "last_hidden_state"):
                        hidden_states = hidden_states.last_hidden_state
                    elif isinstance(hidden_states, dict) and "visible_embeddings" in hidden_states:
                        hidden_states = hidden_states["visible_embeddings"]
                else:
                    raise ValueError(f"Unsupported model_family: {args.model_family}")
        return hidden_states
    
    # Chunking is needed
    device = images.device
    
    # Note: probe_frame_indices contains original video frame indices (for tracking),
    # but we always work with local indices (0 to T-1) within the current batch
    # The images tensor already contains the selected frames in order
    S = T  # Number of frames to process (same as T since images already contains selected frames)
    
    # Use local frame indices (0, 1, 2, ..., T-1) for indexing within current batch
    target_frames = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # [B, T]
    
    # Compute frame-to-chunk mapping (same for all samples in batch)
    frame_to_chunk, num_chunks, chunk_frame_lists = compute_frame_to_chunk_mapping(T, max_context)
    
    # Extract features chunk by chunk
    all_chunk_features = []  # List of [B, chunk_size, num_patches, hidden_dim] or [B, num_video_tokens, hidden_dim]
    
    for chunk_id in range(num_chunks):
        chunk_frames = chunk_frame_lists[chunk_id]  # Frame indices in this chunk
        chunk_size = len(chunk_frames)
        
        # Gather frames for this chunk: [B, chunk_size, C, H, W]
        chunk_images = torch.stack([images[:, idx] for idx in chunk_frames], dim=1)
        
        # Extract features for this chunk
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                if args.model_family in ["clip", "siglip", "siglip2", "dinov2", "dinov3", "metaclip", "llava_vit_si", "aimv2"]:
                    # For image models, process frames individually
                    chunk_frames_flat = chunk_images.view(B * chunk_size, C, H, W)
                    chunk_hidden = model(chunk_frames_flat)
                    if isinstance(chunk_hidden, dict) and "visible_embeddings" in chunk_hidden:
                        chunk_hidden = chunk_hidden["visible_embeddings"]
                    # Reshape to [B, chunk_size, num_patches, hidden_dim]
                    num_patches = chunk_hidden.shape[-2] if len(chunk_hidden.shape) > 2 else 1
                    chunk_hidden = chunk_hidden.view(B, chunk_size, -1, chunk_hidden.size(-1))
                elif args.model_family == "llava_vit_sampling":
                    # For video models with reference frame prepended
                    chunk_videos = chunk_images.transpose(1, 2)  # [B, C, chunk_size, H, W]
                    chunk_hidden = model(chunk_videos, None)
                    if hasattr(chunk_hidden, "last_hidden_state"):
                        chunk_hidden = chunk_hidden.last_hidden_state
                    elif isinstance(chunk_hidden, dict) and "visible_embeddings" in chunk_hidden:
                        chunk_hidden = chunk_hidden["visible_embeddings"]
                    # chunk_hidden: [B, num_video_tokens, hidden_dim]
                    # Need to reshape to per-frame features if possible
                    # For simplicity, we'll store the entire chunk's features
                else:
                    raise ValueError(f"Unsupported model_family: {args.model_family}")
        
        all_chunk_features.append((chunk_id, chunk_frames, chunk_hidden))
    
    # Map features back to target frames using π(t)
    # For each target frame, find its chunk and extract corresponding features
    output_features = []
    
    for batch_idx in range(B):
        batch_features = []
        for frame_idx_in_target in range(S):
            frame_idx = target_frames[batch_idx, frame_idx_in_target].item()
            
            # Get chunk and local position for this frame
            chunk_mappings = frame_to_chunk[frame_idx]
            
            # Use the first occurrence (primary chunk for this frame)
            chunk_id, local_idx = chunk_mappings[0]
            
            # Extract feature from the corresponding chunk
            _, chunk_frames, chunk_hidden = all_chunk_features[chunk_id]
            
            # Get feature for this frame
            if args.model_family in ["clip", "siglip", "siglip2", "dinov2", "dinov3", "metaclip", "llava_vit_si", "aimv2"]:
                # chunk_hidden: [B, chunk_size, num_patches, hidden_dim]
                frame_feat = chunk_hidden[batch_idx, local_idx]  # [num_patches, hidden_dim]
                batch_features.append(frame_feat)
            elif args.model_family == "llava_vit_sampling":
                # For video models, features might be temporally pooled
                # We need to divide the features by chunk size to get per-frame features
                # This is a simplified approach - actual implementation may vary
                num_tokens = chunk_hidden.shape[1]
                tokens_per_frame = num_tokens // len(chunk_frames)
                start_idx = local_idx * tokens_per_frame
                end_idx = start_idx + tokens_per_frame
                frame_feat = chunk_hidden[batch_idx, start_idx:end_idx]  # [tokens_per_frame, hidden_dim]
                batch_features.append(frame_feat)
        
        # Concatenate features for all target frames in this batch
        batch_features = torch.cat(batch_features, dim=0)  # [S*num_patches or S*tokens_per_frame, hidden_dim]
        output_features.append(batch_features)
    
    # Stack batch: [B, total_tokens, hidden_dim]
    output_features = torch.stack(output_features, dim=0)
    
    return output_features


def train_one_experiment(
    args: argparse.Namespace,
    lr: float,
    device: torch.device,
    base_model: nn.Module,
    train_loader,
    val_loader,
) -> tuple:
    """Train pose estimation head with attentive probing."""
    base_model.to(device).eval()
    
    # Create pose estimation head
    head = RelativePoseHead(
        hidden_dim=args.embedding_size,
        pose_hidden_dim=args.pose_hidden_dim
    )
    head.to(device)
    head = torch.nn.parallel.DistributedDataParallel(head, device_ids=[args.local_rank])
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=lr,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=args.default_weight_decay
    )
    
    steps_per_epoch = len(train_loader)
    total_iters = steps_per_epoch * args.default_epoch
    
    scheduler = None
    if args.default_min_lr < lr:
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args.default_min_lr / lr,
            total_iters=total_iters
        )
    
    best_metrics = {"loss": float('inf'), "epe": float('inf'), "rot_error": float('inf'), "trans_error": float('inf')}
    
    for epoch in range(args.default_epoch):
        head.train()
        
        epoch_loss = 0.0
        epoch_epe = 0.0
        epoch_rot_error = 0.0
        epoch_trans_error = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            images = batch["images"].to(device, non_blocking=True)  # [B, T, 3, H, W] where T=num_frames
            target_pose = batch["relative_pose"].to(device, non_blocking=True)  # [B, 12]
            probe_indices = batch["probe_indices"].to(device, non_blocking=True)  # [B, T]
            
            # Extract features (no probe mode in training, use all frames)
            with torch.no_grad():
                feats = get_feature(args, images, base_model, probe_frame_indices=None)
            
            # Predict pose
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred_pose = head(feats)
                loss = pose_loss(pred_pose, target_pose)
            
            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = clip_grad_norm_(head.parameters(), max_norm=args.clip_grad)
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # Compute metrics (apply Procrustes and compute EPE)
            with torch.no_grad():
                # Apply Procrustes to project rotation to SO(3)
                pred_R_mat = pred_pose[:, :9].view(-1, 3, 3).float()
                pred_R_proj = procrustes_rotation(pred_R_mat)
                pred_t = pred_pose[:, 9:].float()
                
                target_R_mat = target_pose[:, :9].view(-1, 3, 3).float()
                target_t = target_pose[:, 9:].float()
                
                # Compute EPE metric
                epe = compute_epe_metric(pred_R_proj, pred_t, target_R_mat, target_t).mean()
                
                # Also compute rotation and translation errors for monitoring
                rot_err = rotation_error(pred_R_proj.view(-1, 9), target_R_mat.view(-1, 9)).mean()
                trans_err = translation_error(pred_t, target_t).mean()
            
            epoch_loss += loss.item()
            epoch_epe += epe.item()
            epoch_rot_error += rot_err.item()
            epoch_trans_error += trans_err.item()
            num_batches += 1
            
            if (i + 1) % args.print_freq == 0 and args.rank == 0:
                elapsed = time.time() - start_time
                samples_per_sec = args.print_freq * args.batch_size * args.world_size / elapsed
                print(
                    f"Epoch: [{epoch}][{i+1}/{steps_per_epoch}]  "
                    f"Speed: {samples_per_sec:.2f} samples/s  "
                    f"Loss: {loss.item():.4f}  "
                    f"EPE: {epe.item():.4f}  "
                    f"Rot Error: {rot_err.item():.2f}°  "
                    f"Trans Error: {trans_err.item():.4f}  "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
                start_time = time.time()
        
        # Epoch metrics
        epoch_loss /= num_batches
        epoch_epe /= num_batches
        epoch_rot_error /= num_batches
        epoch_trans_error /= num_batches
        
        if args.rank == 0:
            print(f"Epoch {epoch} Train - Loss: {epoch_loss:.4f}, EPE: {epoch_epe:.4f}, "
                  f"Rot Error: {epoch_rot_error:.2f}°, Trans Error: {epoch_trans_error:.4f}")
        
        # Evaluation
        if epoch % args.eval_freq == 0 or epoch == args.default_epoch - 1:
            val_metrics = evaluate(args, head, device, base_model, val_loader)
            
            if val_metrics["epe"] < best_metrics["epe"]:
                best_metrics = val_metrics
            
            if args.rank == 0:
                print(f"[Val][Epoch {epoch}] EPE: {val_metrics['epe']:.4f}, "
                      f"Rot Error: {val_metrics['rot_error']:.2f}°, "
                      f"Trans Error: {val_metrics['trans_error']:.4f}")
                print(f"Best - EPE: {best_metrics['epe']:.4f}, "
                      f"Rot Error: {best_metrics['rot_error']:.2f}°, "
                      f"Trans Error: {best_metrics['trans_error']:.4f}")
    
    return best_metrics


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
    head: nn.Module,
    device: torch.device,
    base_model: nn.Module,
    val_loader,
) -> Dict[str, float]:
    """Evaluate pose estimation with EPE metric using probe sampling."""
    head.eval()
    
    total_loss = 0.0
    total_epe = 0.0
    total_rot_error = 0.0
    total_trans_error = 0.0
    num_batches = 0
    
    for batch in val_loader:
        images = batch["images"].to(device, non_blocking=True)  # [B, S, 3, H, W] where S=probe_num_frames
        target_pose = batch["relative_pose"].to(device, non_blocking=True)  # [B, 12]
        probe_indices = batch["probe_indices"].to(device, non_blocking=True)  # [B, S]
        
        # Extract features with probe frame indices
        feats = get_feature(args, images, base_model, probe_frame_indices=probe_indices)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred_pose = head(feats)
            loss = pose_loss(pred_pose, target_pose)
        
        # Apply Procrustes and compute metrics
        pred_R_mat = pred_pose[:, :9].view(-1, 3, 3).float()
        pred_R_proj = procrustes_rotation(pred_R_mat)
        pred_t = pred_pose[:, 9:].float()
        
        target_R_mat = target_pose[:, :9].view(-1, 3, 3).float()
        target_t = target_pose[:, 9:].float()
        
        # Compute EPE metric
        epe = compute_epe_metric(pred_R_proj, pred_t, target_R_mat, target_t).mean()
        rot_err = rotation_error(pred_R_proj.view(-1, 9), target_R_mat.view(-1, 9)).mean()
        trans_err = translation_error(pred_t, target_t).mean()
        
        total_loss += loss.item()
        total_epe += epe.item()
        total_rot_error += rot_err.item()
        total_trans_error += trans_err.item()
        num_batches += 1
    
    metrics = {
        "loss": total_loss / num_batches,
        "epe": total_epe / num_batches,
        "rot_error": total_rot_error / num_batches,
        "trans_error": total_trans_error / num_batches,
    }
    
    return metrics

def get_model(args: argparse.Namespace) -> nn.Module:
    if args.model_name == "hf_llava_vit_large_ln":
        model = AutoModel.from_pretrained(
            args.model_weight,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        return model
    
    model = create_model(args.model_name, pretrained=False)
    if args.model_family in ["llava_vit_sampling"]:
        state_dict = torch.load(args.model_weight, map_location="cpu")
        state_dict = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing, "Unexpected keys:", unexpected)
    return model


def main():
    args = parse_args()
    
    # Setup distributed training
    try:
        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        distributed.init_process_group("nccl")
    except KeyError:
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        distributed.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:12586",
            rank=args.rank,
            world_size=args.world_size
        )
    
    torch.cuda.set_device(args.local_rank)
    device = torch.device(args.local_rank)
    
    if args.rank == 0:
        print("=" * 80)
        print("Attentive Probing: RE10K Relative Pose Estimation")
        print("Following KITTI/ScanNet Training Paradigm")
        print("=" * 80)
        print(f"RE10K Dir: {args.re10k_dir}")
        print(f"Annotation Dir: {args.re10k_annotation_dir}")
        print(f"Model: {args.model_name}")
        print(f"Split: {args.split}")
        print("=" * 80)
    
    # Create dataloaders
    if args.rank == 0:
        print("Creating dataloaders...")
    

    if args.rank == 0:
        print("Using DALI dataloader for optimized multi-GPU performance")
    
    # Training: use normal mode (16 frames for full feature extraction)
    train_loader = get_re10k_dataloader_dali(
        re10k_dir=args.re10k_dir,
        re10k_annotation_dir=args.re10k_annotation_dir,
        split="train",
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.num_workers,
        dali_num_threads=args.dali_num_threads,
        dali_py_num_workers=args.dali_py_num_workers,
        image_cache_size=args.image_cache_size,
        prefetch_queue_depth=args.prefetch_queue_depth,
        py_start_method=args.py_start_method,
        seed=args.seed,
        rank=args.rank,
        probe_mode=False,  # Normal mode for training
        num_frames_to_sample=args.num_frames,
    )
    
    # Validation: use probe mode (S=4 frames with constraints)
    val_loader = get_re10k_dataloader_dali(
        re10k_dir=args.re10k_dir,
        re10k_annotation_dir=args.re10k_annotation_dir,
        split="test",
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.num_workers,
        dali_num_threads=args.dali_num_threads,
        dali_py_num_workers=args.dali_py_num_workers,
        image_cache_size=args.image_cache_size,
        prefetch_queue_depth=args.prefetch_queue_depth,
        py_start_method=args.py_start_method,
        seed=args.seed,
        rank=args.rank,
        probe_mode=True,  # Probe mode for evaluation
        probe_num_frames=args.probe_num_frames,
        probe_min_gap=args.probe_min_gap,
    )

    if args.rank == 0:
        # Print dataset info if available (some dataloaders like DALI Pipeline don't have .dataset)
        if hasattr(train_loader, 'dataset'):
            print(f"Train samples: {len(train_loader.dataset)}")
        else:
            print(f"Train loader created (dataset size not available)")
        
        if hasattr(val_loader, 'dataset'):
            print(f"Val samples: {len(val_loader.dataset)}")
        else:
            print(f"Val loader created (dataset size not available)")
    
    # Load model
    if args.rank == 0:
        print("Loading encoder model...")
    
    base_model = get_model(args)
    
    # Train with different learning rates
    lrs = args.default_lr_list if isinstance(args.default_lr_list, list) else [args.default_lr_list]
    best_lr = 0.0
    best_metrics = {"loss": float('inf'), "epe": float('inf'), "rot_error": float('inf'), "trans_error": float('inf')}
    
    for lr in lrs:
        if args.rank == 0:
            print(f"\nTraining with LR: {lr}")
        
        metrics = train_one_experiment(args, lr, device, base_model, train_loader, val_loader)
        
        if metrics["epe"] < best_metrics["epe"]:
            best_lr = lr
            best_metrics = metrics
    
    if args.rank == 0:
        print("\n" + "=" * 80)
        print("Final Results")
        print("=" * 80)
        print(f"Best LR: {best_lr}")
        print(f"Best EPE: {best_metrics['epe']:.4f}")
        print(f"Best Rotation Error: {best_metrics['rot_error']:.2f}°")
        print(f"Best Translation Error: {best_metrics['trans_error']:.4f}")
        
        # Save report
        save_path = os.path.join(
            args.save_report,
            f"report_attentive_probe_re10k_pose_{os.path.basename(args.model_weight)}.txt"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "w") as f:
            f.write(f"Attentive Probing - RE10K Relative Pose Estimation\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Weight: {args.model_weight}\n")
            f.write(f"Best LR: {best_lr}\n")
            f.write(f"EPE: {best_metrics['epe']:.4f}\n")
            f.write(f"Rotation Error: {best_metrics['rot_error']:.2f}°\n")
            f.write(f"Translation Error: {best_metrics['trans_error']:.4f}\n")
        
        print(f"\nReport saved to: {save_path}")


if __name__ == "__main__":
    main()
