import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

def generate_animation_gif(
    prompt: str,
    output_path: str = "animation.gif",
    num_frames: int = 32,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: int = 0
) -> str:
    """
    Generate an animation GIF using the AnimateDiffPipeline.

    Args:
        prompt (str): The text description for the animation.
        output_path (str): The file path to save the generated GIF (default: "animation.gif").
        num_frames (int): The number of frames in the animation (default: 32).
        guidance_scale (float): Guidance scale for text-to-image generation (default: 7.5).
        num_inference_steps (int): Number of inference steps (default: 50).
        seed (int): Random seed for reproducibility (default: 0).

    Returns:
        str: Path to the saved GIF.
    """
    # Load motion adapter
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-2", 
        torch_dtype=torch.float16
    )

    # Load AnimateDiff pipeline
    pipeline = AnimateDiffPipeline.from_pretrained(
        "emilianJR/epiCRealism", 
        motion_adapter=adapter, 
        torch_dtype=torch.float16
    )

    # Set up scheduler
    scheduler = DDIMScheduler.from_pretrained(
        "emilianJR/epiCRealism",
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipeline.scheduler = scheduler

    # Memory optimizations
    pipeline.enable_vae_slicing()
    pipeline.enable_model_cpu_offload()

    # Generate animation
    output = pipeline(
        prompt=prompt,
        negative_prompt=(
            "Distorted, discontinuous, ugly, blurry, low resolution, motionless, static, "
            "glitchy, unrealistic, oversaturated, noisy, unnatural movements, abrupt transitions, "
            "broken frames, pixelated, misaligned, disjointed, flickering, low-quality textures, "
            "missing details"
        ),
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    
    # Export frames to GIF
    frames = output.frames[0]
    export_to_gif(frames, output_path)
    return output_path
