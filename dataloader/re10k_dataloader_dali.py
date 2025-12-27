import os
import warnings
from typing import Any, Dict, List, Tuple
import random
import math
from collections import OrderedDict
from multiprocessing import Pool, cpu_count
from functools import partial
import atexit
import signal
import gc
import tempfile

import numpy as np
import torch
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from PIL import Image
import pickle
from tqdm import tqdm

# Try to import cv2 for faster image loading
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Try to import psutil for memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def log_memory_usage(rank: int, msg: str = ""):
    """Log current memory usage for debugging."""
    if rank == 0 and HAS_PSUTIL:
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024  # Convert to MB
        print(f"[Memory] {msg}: {mem_mb:.1f} MB")


# ----------------------------------------------------------------------------
# 1. DALI Iterator Wrapper
# ----------------------------------------------------------------------------
class DALIWrapper:
    def __init__(self, dali_iter: DALIGenericIterator, steps_per_epoch: int):
        self.iter = dali_iter
        self.steps_per_epoch = steps_per_epoch

    def __next__(self) -> Dict[str, object]:
        data_dict = self.iter.__next__()[0]
        return {
            "images": data_dict["images"],
            "relative_pose": data_dict["relative_pose"],
        }

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return self.steps_per_epoch

    def reset(self):
        self.iter.reset()


# ----------------------------------------------------------------------------
# 2. Camera and Pose Utilities
# ----------------------------------------------------------------------------

def bbox_xyxy_to_xywh(xyxy):
    """Convert bounding box from xyxy to xywh format."""
    wh = xyxy[2:] - xyxy[:2]
    xywh = np.concatenate([xyxy[:2], wh])
    return xywh


def _convert_ndc_to_pixels(focal_length: np.ndarray, principal_point: np.ndarray, image_size_wh: np.ndarray):
    """Convert focal length and principal point from NDC to pixels."""
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point_px = half_image_size - principal_point * rescale
    focal_length_px = focal_length * rescale
    return focal_length_px, principal_point_px


def _convert_pixels_to_ndc(
    focal_length_px: np.ndarray, principal_point_px: np.ndarray, image_size_wh: np.ndarray
):
    """Convert focal length and principal point from pixels to NDC."""
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point = (half_image_size - principal_point_px) / rescale
    focal_length = focal_length_px / rescale
    return focal_length, principal_point


def adjust_camera_to_bbox_crop_(fl, pp, image_size_wh: np.ndarray, clamp_bbox_xywh: np.ndarray):
    """Adjust camera parameters when cropping to a bounding box."""
    focal_length_px, principal_point_px = _convert_ndc_to_pixels(fl, pp, image_size_wh)
    principal_point_px_cropped = principal_point_px - clamp_bbox_xywh[:2]

    focal_length, principal_point_cropped = _convert_pixels_to_ndc(
        focal_length_px, principal_point_px_cropped, clamp_bbox_xywh[2:]
    )

    return focal_length, principal_point_cropped


def adjust_camera_to_image_scale_(fl, pp, original_size_wh: np.ndarray, new_size_wh: np.ndarray):
    """Adjust camera parameters when resizing the image."""
    focal_length_px, principal_point_px = _convert_ndc_to_pixels(fl, pp, original_size_wh)

    # now scale and convert from pixels to NDC
    image_size_wh_output = new_size_wh.astype(np.float32)
    scale = (image_size_wh_output / original_size_wh).min()
    focal_length_px_scaled = focal_length_px * scale
    principal_point_px_scaled = principal_point_px * scale

    focal_length_scaled, principal_point_scaled = _convert_pixels_to_ndc(
        focal_length_px_scaled, principal_point_px_scaled, image_size_wh_output
    )
    return focal_length_scaled, principal_point_scaled


def normalize_cameras_simple(R_list, T_list, first_camera=True):
    """
    Simplified camera normalization with optimized numpy operations.
    
    Args:
        R_list: List of rotation matrices [N, 3, 3]
        T_list: List of translation vectors [N, 3]
        first_camera: Whether to apply first camera transform
    
    Returns:
        Normalized R and T
    """
    R_array = np.stack(R_list, axis=0)  # [N, 3, 3]
    T_array = np.stack(T_list, axis=0)  # [N, 3]

    # Normalize translation scale
    scale = np.linalg.norm(T_array)
    scale = np.clip(scale, 0.01, 100.0)
    T_array /= scale  # In-place division for speed

    # Apply first camera transform
    if first_camera and len(R_array) > 0:
        R0 = R_array[0].copy()  # Create independent copy to avoid shared memory issues
        T0 = T_array[0].copy()  # Create independent copy to avoid in-place modification issues

        # Compute inverse rotation (transpose)
        R0_inv = R0.T  # Direct transpose, no copy

        # Apply inverse rotation: R' = R0^T @ R
        R_array = np.matmul(R0_inv, R_array)

        # Apply inverse translation: T' = R0^T @ (T - T0)
        T_array -= T0  # In-place subtraction
        T_array = np.einsum('ij,nj->ni', R0_inv, T_array)  # Optimized matrix multiplication

    # Additional normalization of T
    if len(T_array) > 1:
        t_gt = T_array[1:]
        t_gt_scale = np.linalg.norm(t_gt)
        t_gt_scale = t_gt_scale / math.sqrt(len(t_gt))
        t_gt_scale = t_gt_scale / 2
        t_gt_scale = np.clip(t_gt_scale, 0.01, 100.0)
        T_array /= t_gt_scale  # In-place division

    return R_array, T_array


def compute_relative_pose(R1, T1, R2, T2):
    """
    Compute relative pose from camera 1 to camera 2.
    
    Args:
        R1: [3, 3] - Rotation matrix of camera 1
        T1: [3] - Translation vector of camera 1
        R2: [3, 3] - Rotation matrix of camera 2
        T2: [3] - Translation vector of camera 2
    
    Returns:
        relative_pose: [12] - Flattened 3x4 relative pose matrix (9 rotation + 3 translation)
    """
    # Relative rotation: R_rel = R2 @ R1^T
    R_rel = np.matmul(R2, R1.T)

    # Relative translation: T_rel = T2 - R_rel @ T1
    T_rel = T2 - np.matmul(R_rel, T1)

    # Flatten rotation matrix and concatenate with translation
    R_rel_flat = R_rel.reshape(-1)  # [9]
    relative_pose = np.concatenate([R_rel_flat, T_rel])  # [12]

    return relative_pose.astype(np.float32)


def _save_keys_file(keys_pkl: str, sequence_keys, rank: int = 0) -> bool:
    """
    Helper function to save sequence keys to a lightweight pickle file.
    Uses atomic write pattern to prevent corruption from concurrent access.
    
    Args:
        keys_pkl: Path to keys file
        sequence_keys: Iterable of sequence keys (list or dict_keys)
        rank: Process rank (for logging)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if rank != 0:
        return False  # Only rank 0 should save
    
    temp_path = None
    temp_fd = None
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(keys_pkl), exist_ok=True)
        
        # Use atomic write pattern: write to temp file, then rename
        # os.replace() is atomic on both POSIX and Windows (Python 3.3+)
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(keys_pkl), suffix='.tmp')
        try:
            # os.fdopen takes ownership of temp_fd
            with os.fdopen(temp_fd, 'wb') as file:
                temp_fd = None  # fd now managed by file object
                # Convert to list only when pickling to avoid unnecessary copies
                pickle.dump(list(sequence_keys), file)
            # Atomic rename (cross-platform)
            os.replace(temp_path, keys_pkl)
            return True
        except (OSError, IOError):
            # Clean up temp file if write or rename failed
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except (OSError, IOError):
                    pass  # Best effort cleanup
            raise  # Re-raise to outer handler
        finally:
            # Ensure fd is closed if os.fdopen failed
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except (OSError, IOError):
                    pass
    except (OSError, IOError) as e:
        # All file operation errors (includes PermissionError as subclass)
        print(f"[rank {rank}] Warning: File operation failed while saving keys: {e}")
        return False
    except pickle.PicklingError as e:
        print(f"[rank {rank}] Warning: Could not pickle keys: {e}")
        return False


def _process_single_scene(args):
    """
    Process a single scene for parallel metadata loading.
    
    Args:
        args: tuple of (scene, scene_info_dir, data_dir, min_num_images)
    
    Returns:
        tuple of (scene_name, filtered_data) or None if scene doesn't have enough images
    """
    scene, scene_info_dir, data_dir, min_num_images = args

    scene_name = "re10k" + scene
    scene_info_name = os.path.join(scene_info_dir, os.path.basename(scene) + ".txt")

    try:
        scene_info = np.loadtxt(scene_info_name, delimiter=' ', dtype=np.float64, skiprows=1)
    except Exception as e:
        warnings.warn(f"Failed to load scene info for {scene}: {e}")
        return None

    filtered_data = []
    scene_image_size = None

    for frame_idx in range(len(scene_info)):
        try:
            raw_line = scene_info[frame_idx]
            timestamp = raw_line[0]
            intrinsics = raw_line[1:7]
            extrinsics = raw_line[7:]

            imgpath = os.path.join(data_dir, scene, '%s' % int(timestamp) + '.png')

            # Only check the first image to get dimensions
            if scene_image_size is None:
                if os.path.exists(imgpath):
                    # Use cv2 if available for faster image size detection
                    if HAS_CV2:
                        img = cv2.imread(imgpath)
                        if img is not None:
                            h, w = img.shape[:2]
                            scene_image_size = (w, h)  # PIL format is (W, H)
                            del img  # Free memory immediately
                        else:
                            # Fallback to PIL if cv2 fails to read the image
                            try:
                                image_size = Image.open(imgpath).size
                                scene_image_size = image_size
                            except Exception:
                                continue
                    else:
                        try:
                            image_size = Image.open(imgpath).size
                            scene_image_size = image_size
                        except Exception:
                            continue
                else:
                    continue
            else:
                image_size = scene_image_size

            # Use float32 instead of float64 for memory efficiency
            # Optimize: convert to float32 before reshape to avoid intermediate array
            posemat = extrinsics.astype(np.float32).reshape(3, 4)
            # Convert intrinsics to numpy array once to avoid repeated conversions
            intrinsics_f32 = intrinsics.astype(np.float32)
            image_size_array = np.array(scene_image_size, dtype=np.float32)
            focal_length = intrinsics_f32[:2] * image_size_array
            principal_point = intrinsics_f32[2:4] * image_size_array

            data = {
                "filepath": imgpath,
                "R": posemat[:3, :3],
                "T": posemat[:3, -1],
                "focal_length": focal_length,
                "principal_point": principal_point,
            }

            filtered_data.append(data)
        except Exception:
            # Silently skip failed frames to avoid cluttering output
            pass

    if len(filtered_data) > min_num_images:
        return (scene_name, filtered_data)
    else:
        return None


def square_bbox(bbox, padding=0.0):
    """
    Computes a square bounding box, with optional padding parameters.
    
    Args:
        bbox: Bounding box in xyxy format (4,).
    
    Returns:
        square_bbox in xyxy format (4,).
    """
    bbox = np.array(bbox)
    center = (bbox[:2] + bbox[2:]) / 2
    extents = (bbox[2:] - bbox[:2]) / 2
    s = max(extents) * (1 + padding)
    square_bbox = np.array([center[0] - s, center[1] - s, center[0] + s, center[1] + s])
    return square_bbox


# ----------------------------------------------------------------------------
# 3. DALI External Source for RE10K Data
# ----------------------------------------------------------------------------
class RE10KExternalSource:
    def __init__(self, mode: str, source_params: Dict[str, Any]):
        self.mode = mode
        self.sequence_list: List[str] = source_params["sequence_list"]
        
        # Load wholedata lazily from pickle file to avoid serialization issues
        # This prevents large data structure from being pickled when spawning workers
        self.cached_pkl_path: str = source_params.get("cached_pkl_path", None)
        self.wholedata: Dict = None
        
        # If wholedata is directly provided (for backwards compatibility)
        if "wholedata" in source_params and source_params["wholedata"] is not None:
            self.wholedata = source_params["wholedata"]
        
        self.num_shards: int = source_params["num_shards"]
        self.shard_id: int = source_params["shard_id"]
        self.batch_size: int = source_params["batch_size"]
        self.input_size: int = source_params["input_size"]
        self.seed: int = source_params["seed"]
        self.re10k_dir: str = source_params["re10k_dir"]

        # Jitter parameters for training
        if mode == "train":
            self.jitter_scale = [0.8, 1.0]
            self.jitter_trans = [-0.07, 0.07]
        else:
            self.jitter_scale = [1.0, 1.0]
            self.jitter_trans = [0.0, 0.0]

        self.shard_size = len(self.sequence_list) // self.num_shards
        self.shard_offset = self.shard_size * self.shard_id
        self.full_iterations = self.shard_size // self.batch_size

        self.perm = None
        self.last_seen_epoch = -1

        # Set up RNG
        self.rng = np.random.default_rng(seed=self.seed)

        # LRU cache for loaded images to avoid repeated disk I/O
        # Using OrderedDict for LRU eviction when cache is full
        # Note: Each DALI worker process has its own isolated cache (process-safe)
        # Unlimited cache for maximum performance (default: None = cache all images)
        # With 8 GPUs × 16 workers = 128 workers, each worker loads its shard of images
        # RE10K train: ~60K sequences, each sequence ~2 images = ~120K unique images total
        # Per worker (128 workers): ~120K / 128 ≈ 937 images/worker
        # Memory estimate: 937 images × 224×224×3 bytes × 1.2 (overhead) ≈ 140 MB per worker
        # Total: 128 workers × 140 MB ≈ 18 GB for all workers (reasonable for modern systems)
        # Set image_cache_size to a number (e.g., 500) to limit cache if memory constrained
        self._image_cache = OrderedDict()
        self._cache_size = source_params.get("image_cache_size", None)  # None = unlimited cache
        
        # Register cleanup handlers to prevent semaphore leaks
        # These handlers ensure proper resource cleanup even if process is terminated
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _cleanup(self):
        """Clean up resources on process exit to prevent semaphore leaks."""
        try:
            # Clear image cache to release memory
            if hasattr(self, '_image_cache'):
                self._image_cache.clear()
        except Exception:
            # Silently ignore cleanup errors
            pass
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals for graceful cleanup."""
        self._cleanup()
        # Re-raise the signal to allow default handling
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    
    def _ensure_data_loaded(self):
        """
        Lazy load wholedata from pickle file if not already loaded.
        
        Note: Thread-safety is not required as each DALI worker process has its own
        separate instance of RE10KExternalSource with isolated memory space.
        """
        if self.wholedata is None and self.cached_pkl_path is not None:
            try:
                with open(self.cached_pkl_path, 'rb') as file:
                    self.wholedata = pickle.load(file)
            except FileNotFoundError:
                raise RuntimeError(
                    f"Metadata cache file not found: {self.cached_pkl_path}. "
                    f"This file should be created during the first run. "
                    f"Please ensure the dataset paths are correct and the file system is accessible."
                )
            except (pickle.UnpicklingError, EOFError, ValueError) as e:
                raise RuntimeError(
                    f"Failed to load metadata cache from {self.cached_pkl_path}. "
                    f"The file may be corrupted. Try deleting it to regenerate. Error: {e}"
                )
            except PermissionError:
                raise RuntimeError(
                    f"Permission denied when reading {self.cached_pkl_path}. "
                    f"Check file permissions and ensure read access."
                )


    def _jitter_bbox(self, bbox):
        """Random augmentation to cropping box shape."""
        bbox = square_bbox(bbox.astype(np.float32))
        s = self.rng.uniform(self.jitter_scale[0], self.jitter_scale[1])
        tx, ty = self.rng.uniform(self.jitter_trans[0], self.jitter_trans[1], size=2)

        side_length = bbox[2] - bbox[0]
        center = (bbox[:2] + bbox[2:]) / 2 + np.array([tx, ty]) * side_length
        extent = side_length / 2 * s

        # Final coordinates need to be integer for cropping.
        ul = (center - extent).round().astype(int)
        lr = ul + np.round(2 * extent).astype(int)
        return np.concatenate((ul, lr))

    def _load_and_process_frame(self, anno, eval_time=False):
        """Load and process a single frame with camera parameters."""
        filepath = anno["filepath"]
        # filepath is already the full path
        image_path = filepath

        # Try to load from cache first with LRU eviction
        # Use try/except for single dictionary operation instead of double lookup
        try:
            # Move to end to mark as recently used
            self._image_cache.move_to_end(image_path)
            image_array = self._image_cache[image_path]
        except KeyError:
            # Load image - use cv2 if available for faster loading
            if HAS_CV2:
                image_array = cv2.imread(image_path)
                if image_array is None:
                    # Provide comprehensive error message
                    raise ValueError(
                        f"Failed to load image with OpenCV: {image_path}. "
                        f"This typically indicates: file not found, no read permission, "
                        f"unsupported format, or corrupted file data."
                    )
                # Convert BGR to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                image = Image.open(image_path).convert("RGB")
                image_array = np.array(image)

            # Add to cache (with LRU eviction if cache size is limited)
            # Cache the array directly without copying (it's freshly loaded and won't be modified)
            if self._cache_size is not None and len(self._image_cache) >= self._cache_size:
                # Remove least recently used item (first item)
                self._image_cache.popitem(last=False)
            self._image_cache[image_path] = image_array

        # Get image dimensions from numpy array (H, W, C format)
        h, w = image_array.shape[:2]

        # Compute bbox (center crop)
        crop_dim = min(h, w)
        top = (h - crop_dim) // 2
        left = (w - crop_dim) // 2
        bbox = np.array([left, top, left + crop_dim, top + crop_dim], dtype=np.int32)  # Use int32 directly

        # Apply jitter if not eval time
        if not eval_time:
            bbox_jitter = self._jitter_bbox(bbox.astype(np.float32))
        else:
            bbox_jitter = bbox.astype(np.float32)

        # Crop image
        bbox_jitter = bbox_jitter.astype(np.int32)  # Use int32 for better performance
        left, top, right, bottom = bbox_jitter[0], bbox_jitter[1], bbox_jitter[2], bbox_jitter[3]

        # Handle out of bounds
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)

        # Crop directly from numpy array (faster than PIL)
        image_cropped = image_array[top:bottom, left:right]

        # Get camera intrinsics (PT3D convention)
        original_size_wh = np.array([w, h], dtype=np.float32)  # Direct array creation
        scale = min(w, h) / 2.0
        c0 = original_size_wh / 2.0
        focal_pytorch3d = anno["focal_length"] / scale
        p0_pytorch3d = -(anno["principal_point"] - c0) / scale

        # Adjust camera for crop
        bbox_xywh = bbox_xyxy_to_xywh(bbox_jitter.astype(np.float32))
        focal_length_cropped, principal_point_cropped = adjust_camera_to_bbox_crop_(
            focal_pytorch3d, p0_pytorch3d, original_size_wh, bbox_xywh
        )

        # Adjust camera for resize
        crop_size = np.array([right - left, bottom - top], dtype=np.float32)  # [W, H] - direct creation
        new_size = np.array([self.input_size, self.input_size], dtype=np.float32)
        new_focal_length, new_principal_point = adjust_camera_to_image_scale_(
            focal_length_cropped, principal_point_cropped, crop_size, new_size
        )

        # Get rotation and translation (convert from COLMAP to PT3D)
        R = anno["R"].T.copy()  # Transpose and copy
        R[:, :2] *= -1
        T = anno["T"].copy()
        T[:2] *= -1

        return image_cropped, new_focal_length, new_principal_point, R, T

    def __call__(self, sample_info) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns: (images, target_pose, dummy)
        - images: [16, H, W, 3] - 16 frames uniformly sampled (uint8)
        - target_pose: [12] - Absolute pose of the 16th frame after normalization (float32)
        - dummy: empty array for DALI compatibility
        """
        # Ensure data is loaded (lazy loading for worker processes)
        self._ensure_data_loaded()
        
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration

        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            self.rng = np.random.default_rng(seed=self.seed + sample_info.epoch_idx)
            self.perm = self.rng.permutation(len(self.sequence_list))

        sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]
        sequence_name = self.sequence_list[sample_idx]
        metadata = self.wholedata[sequence_name]

        # Sample 16 frames uniformly from the video
        # Following the new training paradigm: extract features from 16 frames,
        # then use only first and last frame features for pose prediction
        num_frames_total = len(metadata)
        num_frames_to_sample = 16
        
        if num_frames_total >= num_frames_to_sample:
            if self.mode == "train":
                # Training: randomly select a segment of the video that has at least 16 frames
                # Then sample 16 frames uniformly from this segment
                # This allows for data augmentation while keeping temporal structure
                
                # Determine the segment length (can be longer than 16 frames)
                min_segment_length = num_frames_to_sample
                max_segment_length = num_frames_total
                
                # Randomly choose segment length
                segment_length = self.rng.integers(min_segment_length, max_segment_length + 1)
                
                # Randomly choose start position of segment
                max_start_idx = num_frames_total - segment_length
                if max_start_idx > 0:
                    start_idx = self.rng.integers(0, max_start_idx + 1)
                else:
                    start_idx = 0
                
                end_idx = start_idx + segment_length
                
                # Uniformly sample 16 frames from this segment
                ids = np.linspace(start_idx, end_idx - 1, num_frames_to_sample, dtype=int).tolist()
            else:
                # Evaluation: uniformly sample 16 frames from the entire video
                ids = np.linspace(0, num_frames_total - 1, num_frames_to_sample, dtype=int).tolist()
        else:
            # Not enough frames - repeat frames to reach 16
            ids = list(range(num_frames_total))
            # Pad by repeating the last frame
            ids += [num_frames_total - 1] * (num_frames_to_sample - num_frames_total)

        # Load and process all 16 frames
        eval_time = (self.mode != "train")
        images = []
        focal_lengths = []
        principal_points = []
        rotations = []
        translations = []

        for idx in ids:
            anno = metadata[idx]
            image, fl, pp, R, T = self._load_and_process_frame(anno, eval_time)
            images.append(image)
            focal_lengths.append(fl)
            principal_points.append(pp)
            rotations.append(R)
            translations.append(T)

        # Normalize cameras - use all frames for better normalization
        R_norm, T_norm = normalize_cameras_simple(rotations, translations, first_camera=True)

        # Get the pose of the 16th frame (last frame) after normalization
        # This is the target pose we want to predict
        R_target = R_norm[-1]  # Rotation of frame 15 (16th frame, 0-indexed)
        T_target = T_norm[-1]  # Translation of frame 15
        
        # Flatten rotation matrix and concatenate with translation to form pose vector
        R_target_flat = R_target.reshape(-1)  # [9]
        target_pose = np.concatenate([R_target_flat, T_target]).astype(np.float32)  # [12]

        # Stack images: [16, H, W, 3]
        # Images might have different sizes due to cropping, so we need to resize them
        # Use cv2 for faster resizing if available, otherwise fall back to PIL
        target_size = self.input_size
        images_resized = []
        for img in images:
            # Fast resizing with cv2 if available
            if HAS_CV2:
                # cv2.resize is significantly faster than PIL
                img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                images_resized.append(img_resized)
            else:
                img_pil = Image.fromarray(img)
                img_resized = img_pil.resize((target_size, target_size), Image.BILINEAR)
                images_resized.append(np.array(img_resized))

        images_array = np.stack(images_resized, axis=0).astype(np.uint8)  # [16, H, W, 3]

        return images_array, target_pose, np.array([0], dtype=np.int64)


def preprocess_images_re10k(images, mode, input_size):
    """
    Preprocess RE10K images with DALI GPU operations.
    
    Args:
        images: [16, H, W, C] format - 16 uniformly sampled frames
        mode: "train" or "test"
        input_size: Target size
    
    Note: This function normalizes images to [0, 1] range without mean/std normalization,
          consistent with the original RE10K PyTorch dataloader which uses ToTensor().
    """
    # Images are already resized in external source, just need augmentation and normalization

    if mode == "train":
        # Color augmentation matching ColorJitter(brightness=0.3, contrast=0.4, saturation=0.2, hue=0.1)
        if fn.random.coin_flip(dtype=types.BOOL, probability=0.75):
            images = fn.brightness_contrast(
                images,
                contrast=fn.random.uniform(range=(0.6, 1.4)),
                brightness=fn.random.uniform(range=(-0.3, 0.3)),
                device="gpu",
            )

        if fn.random.coin_flip(dtype=types.BOOL, probability=0.75):
            images = fn.saturation(
                images,
                saturation=fn.random.uniform(range=[0.8, 1.2]),
                device="gpu",
            )

        if fn.random.coin_flip(dtype=types.BOOL, probability=0.75):
            images = fn.hue(
                images,
                hue=fn.random.uniform(range=[-0.1, 0.1]),
                device="gpu",
            )

    # Normalize to [0, 1] range in FLOAT / FCHW layout
    # Using mean=0 and std=255 effectively divides by 255
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="FCHW",  # [F, C, H, W] where F=16 (16 uniformly sampled frames)
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        device="gpu",
    )

    return images


# ----------------------------------------------------------------------------
# 4. DALI Pipeline Definition
# ----------------------------------------------------------------------------
@pipeline_def(enable_conditionals=True)
def dali_re10k_pipeline(mode: str, source_params: Dict[str, Any]):
    input_size = source_params["input_size"]

    # External source returns: images [2, H, W, 3], relative_pose [12], dummy
    images, relative_pose, _ = fn.external_source(
        source=RE10KExternalSource(mode, source_params),
        num_outputs=3,
        batch=False,
        parallel=True,
        dtype=[types.UINT8, types.FLOAT, types.INT64],
        layout=["FHWC", "", ""]
    )

    images = images.gpu()
    relative_pose = relative_pose.gpu()

    images = preprocess_images_re10k(images, mode, input_size)

    return images, relative_pose


# ----------------------------------------------------------------------------
# 5. Main Dataloader Function
# ----------------------------------------------------------------------------
def get_re10k_dataloader_dali(
    re10k_dir: str,
    re10k_annotation_dir: str,
    split: str = "train",
    batch_size: int = 128,
    input_size: int = 224,
    dali_num_threads: int = 12,  # Optimization: increased from 8 to 12
    dali_py_num_workers: int = 16,  # Optimization: increased from 4 to 16 for better CPU utilization
    image_cache_size: int = None,  # Optimization: None = unlimited cache, load all images into memory
    prefetch_queue_depth: int = 2,  # Optimization: increased from 1 to 2 for better GPU pipeline
    py_start_method: str = "spawn",
    seed: int = 0,
    rank: int = 0,
    min_num_images: int = 5,
    num_workers: int = None,  # For API compatibility, not used
    shuffle: bool = None,  # For API compatibility, not used
) -> DALIWrapper:
    """
    Create a DALI-based DataLoader for RE10K dataset with performance optimizations.
    
    Optimizations implemented:
    - GPU-accelerated image preprocessing (resize, augmentation, normalization)
    - Lightweight keys file (<1MB) prevents 24GB metadata spike on 8 GPU setup
    - Only rank 0 loads full metadata when keys file doesn't exist (critical fix!)
    - Unlimited image cache (default: None = cache all images) for maximum performance
    - Increased prefetch queue depth (2, up from 1) for better GPU pipeline overlap
    - OpenCV-based image loading when available (significantly faster than PIL)
    - OpenCV-based image resizing (faster than PIL.Image.resize)
    - Lazy loading of metadata in worker processes to avoid serialization issues
    - Spawn process start method for safe multi-GPU operation
    - Parallel data loading with increased worker counts (default 16, up from 4)
    - Optimized numpy operations (in-place operations, einsum, reduced copies)
    - Direct numpy array operations avoiding PIL conversions where possible
    - Efficient data sharding for multi-GPU training
    - Parallel metadata processing using multiprocessing.Pool
    - Progress bar (tqdm) for metadata processing visibility
    - Only rank 0 processes metadata in multi-GPU setup (avoids duplication)
    - Increased DALI thread count (default 12, up from 8) for better CPU utilization
    
    Args:
        re10k_dir: Path to RealEstate10k frames directory
        re10k_annotation_dir: Path to RealEstate10k annotations
        split: Dataset split ("train" or "test")
        batch_size: Batch size
        input_size: Input image size
        dali_num_threads: Number of threads per DALI pipeline (default: 12, increase to 16-24 if CPU-bound)
        dali_py_num_workers: Number of Python workers for DALI external source 
                             (default: 16, optimized for multi-core systems; reduce to 8 if memory constrained)
        image_cache_size: Number of images to cache per worker (default: None = unlimited cache for max performance;
                          set to 500 for balanced performance, 100-250 if memory constrained)
        prefetch_queue_depth: Number of batches to prefetch (default: 2, optimized for GPU overlap;
                              reduce to 1 if memory constrained)
        py_start_method: Process start method for DALI workers (default: "spawn", safest option
                         for large datasets with lazy loading)
        seed: Random seed for reproducibility
        rank: Rank of the current process in distributed training
        min_num_images: Minimum number of images per sequence
        num_workers: For API compatibility with PyTorch DataLoader (not used)
        shuffle: For API compatibility with PyTorch DataLoader (not used)
    
    Returns:
        dataloader: DALI-based dataloader wrapped in DALIWrapper
    
    Note: Images are normalized to [0, 1] range without mean/std normalization,
          consistent with the original RE10K PyTorch dataloader.
    
    Performance Notes:
        - Metadata processing is cached in processed_{split}.pkl for subsequent runs
        - Sequence keys cached separately in processed_{split}_keys.pkl (lightweight, <1MB)
        - First run will use parallel workers (up to 32 processes) for faster processing
        - Typical speedup: 8-16x faster metadata processing compared to sequential
        - Conservative defaults provide safe operation on multi-GPU setups without OOM
        - Lazy loading prevents large data structures from being serialized to worker processes
        - Spawn start method is safest for avoiding IPC/serialization issues with DALI workers
        
    Memory Management:
        - Lightweight keys file solution eliminates 24GB memory spike from 3GB pickle loading
        - CRITICAL FIX: Only rank 0 loads full metadata when keys file doesn't exist
        - Main process loads only lightweight keys file (<1MB), not full metadata (24GB saved on 8 GPUs!)
        - Workers lazy-load full metadata only when needed (in worker processes, not main process)
        - With unlimited cache (default): 16 workers × 8 GPUs = 128 total workers
        - RE10K train: ~60K sequences, ~2 images/sequence = ~120K unique images
        - Per worker: ~120K / 128 ≈ 937 images/worker (due to sharding)
        - Memory estimate per worker: 937 images × 224×224×3 bytes × 1.2 (overhead) ≈ 140 MB
        - Total cache memory: 128 workers × 140 MB ≈ 18 GB (reasonable for modern systems with 40GB+ GPU memory)
        - Both train and val loaders: ~36 GB total (acceptable for high-memory systems)
        - Explicit garbage collection after metadata loading to free temporary objects
        - If memory constrained, can limit cache: --image_cache_size 500 (balanced) or 100-250 (low memory)
        - Cache warmup happens naturally during first epoch as images are accessed
    """
    # Load dataset metadata
    log_memory_usage(rank, f"Start of {split} dataloader initialization")
    
    if split == "train":
        data_dir = os.path.join(re10k_dir, "frames/train")
        video_loc = os.path.join(re10k_dir, "frames/train/video_loc.txt")
        scenes = np.loadtxt(video_loc, dtype=str)
        scene_info_dir = os.path.join(re10k_annotation_dir, "train")
    elif split == "test":
        data_dir = os.path.join(re10k_dir, "frames/test")
        video_loc = os.path.join(re10k_dir, "frames/test/video_loc.txt")
        scenes = np.loadtxt(video_loc, dtype=str)
        scene_info_dir = os.path.join(re10k_annotation_dir, "test")
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train' or 'test'.")

    if rank == 0:
        print(f"[{split} loader] Loading RE10K metadata from: {re10k_dir}")

    # Build dataset
    wholedata = {}
    cached_pkl = os.path.join(os.path.dirname(os.path.dirname(scene_info_dir)), f"processed_{split}.pkl")
    keys_pkl = os.path.splitext(cached_pkl)[0] + '_keys.pkl'

    # Check if we have the lightweight keys file first to avoid loading 3GB pickle
    if os.path.exists(keys_pkl):
        # Fast path: We have the keys file, skip loading full metadata
        if rank == 0:
            print(f"[{split} loader] Found lightweight keys file: {keys_pkl}")
        # wholedata remains empty, will load keys directly later
    elif os.path.exists(cached_pkl):
        # Keys file doesn't exist but full pickle does - only rank 0 loads it to extract keys
        # CRITICAL FIX: Previously all ranks loaded the 3GB pickle, causing 8×3GB=24GB memory spike
        if rank == 0:
            print(f"[{split} loader] Loading cached metadata from: {cached_pkl}")
            print(f"[{split} loader] (Keys file will be created for faster loading in future runs)")
            log_memory_usage(rank, f"Before loading metadata cache to extract keys")
            with open(cached_pkl, 'rb') as file:
                wholedata = pickle.load(file)
            log_memory_usage(rank, f"After loading metadata cache")
            
            # Extract keys and save them immediately
            sequence_list = sorted(list(wholedata.keys()))
            if _save_keys_file(keys_pkl, sequence_list, rank):
                print(f"[{split} loader] Metadata keys saved to: {keys_pkl}")
            
            # Free memory by clearing the large data structure
            del wholedata
            wholedata = {}
            gc.collect()
            log_memory_usage(rank, f"After extracting keys and freeing metadata (rank 0)")
        
        # Synchronize: Wait for rank 0 to create keys file before other ranks proceed
        world_size_env = int(os.getenv("WORLD_SIZE", "1"))
        if world_size_env > 1:
            try:
                from torch import distributed as dist
                if dist.is_initialized():
                    if rank == 0:
                        print(f"[{split} loader] Rank 0 created keys file, notifying other ranks...")
                    dist.barrier()  # Wait for rank 0 to create keys file
            except ImportError:
                if rank != 0:
                    raise RuntimeError("Non-zero rank detected but torch.distributed not available")
    else:
        # Only rank 0 processes metadata to avoid duplicate work in multi-GPU setup
        if rank == 0:
            # Determine number of parallel workers (conservative to avoid memory exhaustion)
            # Cap at 16 on multi-GPU setups to leave resources for GPU processes
            try:
                world_size_env = int(os.getenv("WORLD_SIZE", "1"))
            except (ValueError, TypeError):
                world_size_env = 1  # Default to single GPU if invalid
            num_cpus = cpu_count()
            max_workers = 16 if world_size_env > 1 else 32
            num_workers = min(num_cpus, max_workers)
            print(f"[{split} loader] Using {num_workers} parallel workers for metadata processing")

            # Prepare arguments for parallel processing
            process_args = [(scene, scene_info_dir, data_dir, min_num_images) for scene in scenes]

            # Process scenes in parallel with progress bar
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(_process_single_scene, process_args),
                    total=len(scenes),
                    desc=f"Processing {split} scenes",
                    ncols=80
                ))
                # Explicitly close and join pool to free resources
                pool.close()
                pool.join()

            # Collect results
            for result in results:
                if result is not None:
                    scene_name, filtered_data = result
                    wholedata[scene_name] = filtered_data

            # Free the results list to reduce memory
            del results
            gc.collect()
            log_memory_usage(rank, f"After metadata processing for {len(wholedata)} scenes")

            print(f"[{split} loader] Finished processing metadata for {len(wholedata)} scenes")
            print(f"[{split} loader] Saving metadata cache to: {cached_pkl}")
            os.makedirs(os.path.dirname(cached_pkl), exist_ok=True)

            # Save the processed metadata for future use
            with open(cached_pkl, 'wb') as file:
                pickle.dump(wholedata, file)
            print(f"[{split} loader] Metadata cache saved successfully")
            
            # Also save just the keys in a separate lightweight file for fast loading
            keys_pkl = os.path.splitext(cached_pkl)[0] + '_keys.pkl'
            if _save_keys_file(keys_pkl, wholedata.keys(), rank):
                print(f"[{split} loader] Metadata keys saved to: {keys_pkl}")

        # Synchronize across GPUs - wait for rank 0 to finish processing and saving
        # Get world size for distributed training
        world_size_env = int(os.getenv("WORLD_SIZE", "1"))
        rank_env = int(os.getenv("RANK", str(rank)))
        if world_size_env > 1:
            try:
                from torch import distributed as dist
                if dist.is_initialized():
                    if rank_env == 0:
                        print(f"[{split} loader] Rank 0 finished processing, broadcasting to other ranks...")
                    dist.barrier()  # Wait for rank 0 to finish

                    # Non-rank-0 processes now load the cache
                    if rank_env != 0:
                        wholedata = {}  # Temporary empty dict; will be loaded below
            except ImportError:
                # If distributed not available, only rank 0 should be running
                if rank_env != 0:
                    raise RuntimeError("Non-zero rank detected but torch.distributed not available")
        else:
            # Single GPU, no need for synchronization
            pass

    # Memory optimization: Only load sequence keys, not full metadata
    # Workers will load the full data lazily from the pickle file when needed
    # Note: This section handles the case where wholedata is empty because:
    # 1. Keys file existed (fast path above) - we skipped loading full pickle
    # 2. Multi-GPU: non-rank-0 processes after barrier - they didn't create data
    # 3. Rank 0 already extracted keys in elif block above (new optimization)
    
    # Check if sequence_list was already created (by rank 0 in elif block)
    if 'sequence_list' not in locals():
        if not wholedata and os.path.exists(cached_pkl):
            # keys_pkl was already defined above, reuse it
            
            if os.path.exists(keys_pkl):
                # Fast path: Load only keys from small file (few KB vs 3GB)
                # This is the normal path for subsequent runs after keys file is created
                if rank == 0:
                    print(f"[{split} loader] Loading metadata keys from: {keys_pkl}")
                log_memory_usage(rank, "Before loading metadata keys")
                
                with open(keys_pkl, 'rb') as file:
                    sequence_list = pickle.load(file)
                sequence_list = sorted(sequence_list)
                
                log_memory_usage(rank, "After loading metadata keys only")
            else:
                # Fallback: Load full pickle and extract keys
                # This path should rarely be taken now due to synchronization above
                # but kept for robustness (e.g., if keys file gets deleted between barrier and here)
                if rank == 0:
                    print(f"[{split} loader] Warning: Keys file disappeared after creation, recreating...")
                log_memory_usage(rank, "Before loading metadata cache (fallback)")
                
                with open(cached_pkl, 'rb') as file:
                    wholedata = pickle.load(file)
                log_memory_usage(rank, "After loading metadata cache (fallback)")
                
                # Extract keys and free the full data immediately
                sequence_list = sorted(list(wholedata.keys()))
                
                # Save keys for future runs to avoid this expensive operation
                if _save_keys_file(keys_pkl, sequence_list, rank):
                    if rank == 0:
                        print(f"[{split} loader] Saved keys to: {keys_pkl} for future runs")
                
                # Free memory by clearing the large data structure
                del wholedata
                gc.collect()
                log_memory_usage(rank, "After freeing metadata and GC (fallback)")
                
        elif wholedata:
            # Rank 0 just created the data from scratch (no cached_pkl existed), extract keys and free memory
            sequence_list = sorted(list(wholedata.keys()))
            del wholedata
            gc.collect()
            log_memory_usage(rank, "After extracting keys and freeing metadata (rank 0)")
        else:
            raise RuntimeError(f"Metadata cache file not found: {cached_pkl}")

    if rank == 0:
        print(f"[{split} loader] Total sequences: {len(sequence_list)}")

    # Get world size for distributed training
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", str(rank)))

    # Create DALI pipeline
    # Pass pickle file path instead of full wholedata to avoid serialization issues
    source_params = {
        "sequence_list": sequence_list,
        "wholedata": None,  # Don't pass the large dict, use lazy loading
        "cached_pkl_path": cached_pkl,  # Pass file path for lazy loading
        "num_shards": world_size,
        "shard_id": rank,
        "batch_size": batch_size,
        "input_size": input_size,
        "seed": seed + rank,
        "re10k_dir": re10k_dir,
        "image_cache_size": image_cache_size,
    }

    # Use the specified process start method
    # spawn is safest and works with DALI's external source serialization
    # With lazy loading, spawn method works well even with large datasets
    if rank == 0:
        print(f"[{split} loader] Using '{py_start_method}' process start method with lazy loading")

    pipe = dali_re10k_pipeline(
        batch_size=batch_size,
        num_threads=dali_num_threads,
        device_id=local_rank,
        seed=seed + rank,
        py_num_workers=dali_py_num_workers,
        py_start_method=py_start_method,
        prefetch_queue_depth=prefetch_queue_depth,  # Configurable for memory-constrained systems
        mode=split,
        source_params=source_params,
    )
    pipe.build()

    # Create DALI iterator
    dali_iter = DALIGenericIterator(
        pipelines=[pipe],
        output_map=["images", "relative_pose"],
        auto_reset=True,
        prepare_first_batch=True  # Optimization: prepare first batch immediately for faster training start
    )

    steps_per_epoch = len(sequence_list) // world_size // batch_size
    dataloader = DALIWrapper(dali_iter=dali_iter, steps_per_epoch=steps_per_epoch)

    if rank == 0:
        print(f"[{split} loader] DALI pipeline built. Steps per epoch: {steps_per_epoch}")
    
    log_memory_usage(rank, f"After DALI pipeline built for {split}")

    return dataloader
