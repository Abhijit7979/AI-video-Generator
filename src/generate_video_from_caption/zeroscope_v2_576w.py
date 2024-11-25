import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

def generate_video_from_caption(caption: str, num_frames: int = 32) -> str:
    """
    Generate a video based on a given text caption using the ZeroScope model.

    Args:
        caption (str): The text description to generate the video.
        num_frames (int): The number of frames in the video (default: 32).

    Returns:
        str: Path to the generated video.
    """
    # Load the pretrained pipeline with memory optimization
    pipe = DiffusionPipeline.from_pretrained(
        "cerspense/zeroscope_v2_576w", 
        torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()
    pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
    pipe.enable_vae_slicing()

    # Generate video frames
    video_frames = pipe(caption, num_frames=num_frames).frames[0]

    # Export frames to a video file
    video_path = export_to_video(video_frames)
    return video_path
