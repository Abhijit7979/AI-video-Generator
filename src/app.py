import gradio as gr
from generate_video_caption.model1 import generate_video_caption  # Video captioning function
from generate_video_from_caption.animatediff_v1_5_2 import generate_animation_gif  # Animation generation function
from IPython.display import Video

def process_video(input_video):
    print(f"Input video path: {input_video}")
    caption = generate_video_caption(input_video)
    print(f"Generated caption: {caption}")
    ai_video_path = generate_animation_gif(
        prompt=caption,
        output_path="generated_animation.gif",
        num_frames=32,
        guidance_scale=7.5,
        num_inference_steps=50,
        seed=0
    )
    print(f"AI-generated video path: {ai_video_path}")
    return input_video, ai_video_path, caption

# Define the Gradio Interface
def interface():
    # Video input widget
    with gr.Blocks() as app:
        gr.Markdown("### AI Video Captioning and Generation")
        with gr.Row():
            input_video = gr.Video(label="Upload Video")
        with gr.Row():
            output_video = gr.Video(label="AI-Generated Video")
            original_video = gr.Video(label="Original Video")
        caption_box = gr.Textbox(label="Generated Caption", lines=2)
        
        process_btn = gr.Button("Generate AI Video")
        
        # Connect the button to the process_video function
        process_btn.click(process_video, inputs=[input_video], outputs=[original_video, output_video, caption_box])
        
    return app

# Launch the app
if __name__ == "__main__":
    interface().launch(debug=True)
