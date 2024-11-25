import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

def generate_video_from_caption_damo(caption: str, num_frames: int = 32) -> str:
    """
    Generate a video based on a given text caption using the Damo-Vilab model.

    Args:
        caption (str): The text description for the video.
        num_frames (int): The number of frames to generate (default: 32).

    Returns:
        str: Path to the generated video.
    """
    # Load the pretrained pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.enable_model_cpu_offload()

    # Memory optimization
    pipe.enable_vae_slicing()

    # Generate video frames
    video_frames = pipe(caption, num_frames=num_frames).frames[0]

    # Export frames to a video file
    video_path = export_to_video(video_frames)
    return video_path
