import streamlit as st
import fitz  # PyMuPDF
import base64
import os
from groq import Groq
from moviepy.editor import ImageClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile

# Initialize Groq client
client = Groq(
    api_key="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"
)

def extract_text_sections(pdf_path):
    """Extract abstract and conclusion from PDF"""
    doc = fitz.open(pdf_path)
    text = ""
    abstract = ""
    conclusion = ""
    
    # Simple heuristic to find abstract and conclusion
    for page in doc:
        text += page.get_text()
    
    # Find abstract
    abstract_start = text.lower().find("abstract")
    if abstract_start != -1:
        abstract_end = text.lower().find("introduction", abstract_start)
        if abstract_end != -1:
            abstract = text[abstract_start:abstract_end].strip()
    
    # Find conclusion
    conclusion_start = text.lower().find("conclusion")
    if conclusion_start != -1:
        conclusion_end = text.lower().find("references", conclusion_start)
        if conclusion_end != -1:
            conclusion = text[conclusion_start:conclusion_end].strip()
        else:
            conclusion = text[conclusion_start:].strip()
    
    return abstract, conclusion

def extract_images_from_pdf(pdf_path, output_folder):
    """Extract images from PDF"""
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{output_folder}/page_{page_num+1}_img_{img_index+1}.{image_ext}"
            
            with open(image_filename, "wb") as f:
                f.write(image_data)
            
            image_paths.append(image_filename)
    
    return image_paths

def get_image_explanation(image_path, context=""):
    """Get explanation for image using Groq API"""
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    image_data_url = f"data:image/png;base64,{image_base64}"
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Explain this figure from a research paper. {context}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    }
                }
            ]
        }
    ]
    
    completion = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=messages,
        temperature=0.7
    )
    
    return completion.choices[0].message.content

def create_text_image(text, size=(800, 150), fontsize=24, color='white', bg_color=(0, 0, 0, 0)):
    """Create text image using PIL and save it as a temporary file."""
    img = Image.new('RGBA', size, bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
    except:
        font = ImageFont.load_default()

    # Word wrap text
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        current_line.append(word)
        test_line = ' '.join(current_line)
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] > size[0] - 20:
            if len(current_line) > 1:
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
            else:
                lines.append(test_line)
                current_line = []

    if current_line:
        lines.append(' '.join(current_line))

    y = 10
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (size[0] - text_width) // 2
        draw.text((x, y), line, font=font, fill=color)
        y += fontsize + 5

    # Save image as a temporary file and return its path
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img.save(temp_file.name)
    return temp_file.name  # Return the file path instead of NumPy array

def create_video_segments(text, title, temp_dir):
    """Create video segment for text sections"""
    tts = gTTS(text=text, lang="en")
    audio_file = f"{temp_dir}/{title.lower()}_audio.mp3"
    tts.save(audio_file)

    audio_clip = AudioFileClip(audio_file)
    duration = audio_clip.duration

    # Create text image and get file path
    text_image_path = create_text_image(text, size=(800, 600), fontsize=32)
    text_clip = ImageClip(text_image_path).set_duration(duration)

    # Create title image and get file path
    title_image_path = create_text_image(title, size=(800, 100), fontsize=48)
    title_clip = ImageClip(title_image_path).set_duration(duration).set_position(("center", 50))

    final_clip = CompositeVideoClip([text_clip, title_clip], size=(800, 600)).set_audio(audio_clip)
    
    # Cleanup temporary files
    os.remove(text_image_path)
    os.remove(title_image_path)

    return final_clip

def create_full_video(abstract, images_with_explanations, conclusion, output_path):
    """Create full video with all segments"""
    clips = []
    temp_dir = tempfile.mkdtemp()
    
    # Add abstract
    if abstract:
        abstract_clip = create_video_segments(abstract, "Abstract", temp_dir)
        clips.append(abstract_clip)
    
    # Add image explanations
    for img_path, explanation in images_with_explanations:
        tts = gTTS(text=explanation, lang="en")
        audio_file = f"{temp_dir}/img_{os.path.basename(img_path)}_audio.mp3"
        tts.save(audio_file)
        
        audio_clip = AudioFileClip(audio_file)
        duration = audio_clip.duration
        
        image_clip = ImageClip(img_path).set_duration(duration).resize(height=400).set_position("center")
        text_image = create_text_image(explanation, size=(800, 150), fontsize=24)
        text_clip = ImageClip(text_image).set_duration(duration).set_position(("center", 450))
        
        final_clip = CompositeVideoClip([image_clip, text_clip], size=(800, 600)).set_audio(audio_clip)
        clips.append(final_clip)
    
    # Add conclusion
    if conclusion:
        conclusion_clip = create_video_segments(conclusion, "Conclusion", temp_dir)
        clips.append(conclusion_clip)
    
    # Concatenate all clips
    final_video = concatenate_videoclips(clips, method="compose")
    final_video.write_videofile(output_path, fps=24)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

def main():
    st.title("Research Paper Video Generator")
    st.write("Upload a research paper PDF to generate an explanatory video.")
    
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded PDF
            pdf_path = os.path.join(temp_dir, "paper.pdf")
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Extract text sections
            abstract, conclusion = extract_text_sections(pdf_path)
            
            # Extract images
            images_folder = os.path.join(temp_dir, "images")
            image_paths = extract_images_from_pdf(pdf_path, images_folder)
            
            if not image_paths:
                st.error("No images found in the PDF.")
                return
            
            st.write(f"Found {len(image_paths)} images in the PDF.")
            
            if st.button("Generate Video"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Get explanations for each image
                images_with_explanations = []
                for i, img_path in enumerate(image_paths):
                    status_text.text(f"Analyzing image {i+1}/{len(image_paths)}...")
                    explanation = get_image_explanation(img_path)
                    images_with_explanations.append((img_path, explanation))
                    progress_bar.progress((i + 1) / (len(image_paths) + 2))
                
                # Create video
                status_text.text("Generating video...")
                output_path = os.path.join(temp_dir, "research_explanation.mp4")
                create_full_video(abstract, images_with_explanations, conclusion, output_path)
                
                # Offer video download
                with open(output_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Video",
                        data=file,
                        file_name="research_explanation.mp4",
                        mime="video/mp4"
                    )
                
                status_text.text("Video generation complete!")
                progress_bar.progress(1.0)

if __name__ == "__main__":
    main()
