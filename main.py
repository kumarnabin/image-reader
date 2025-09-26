import os
import re
import warnings
from flask import Flask, request, render_template, jsonify, send_from_directory
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
# Load BLIP large model (more accurate)
MODEL_NAME = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
def generate_detailed_caption(image: Image.Image) -> dict:
    """Generate multiple detailed captions using different prompts and parameters."""
    # Different prompting strategies for more detailed captions
    prompts = [
        "a detailed description of",
        "this image shows",
        "the scene contains",
        ""  # unconditional generation
    ]
    captions = []
    for prompt in prompts:
        if prompt:
            # Conditional generation with prompt
            inputs = processor(image, text=prompt, return_tensors="pt").to(device)
        else:
            # Unconditional generation
            inputs = processor(image, return_tensors="pt").to(device)
        # Generate with different parameters for variety
        output_ids = model.generate(
            **inputs,
            max_length=100,  # Increased for more detail
            min_length=20,  # Ensure minimum detail
            num_beams=8,  # More beams for better quality
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            length_penalty=1.2  # Encourage longer captions
        )
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        # Remove the prompt from the beginning if it was used
        if prompt and caption.lower().startswith(prompt.lower()):
            caption = caption[len(prompt):].strip()
        captions.append(clean_caption(caption))
    # Select the most detailed caption
    detailed_caption = max(captions, key=len)
    # Generate a short caption as well
    inputs = processor(image, return_tensors="pt").to(device)
    short_output = model.generate(
        **inputs,
        max_length=30,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    short_caption = processor.decode(short_output[0], skip_special_tokens=True)
    short_caption = clean_caption(short_caption)
    return {
        "detailed": detailed_caption,
        "short": short_caption,
        "alternatives": [clean_caption(cap) for cap in captions if cap != detailed_caption]
    }
def clean_caption(caption: str) -> str:
    """Clean and improve caption text."""
    # Remove repeated words and phrases
    caption = re.sub(r'\b(\w+)( \1\b)+', r'\1', caption)
    # Remove repeated phrases
    words = caption.split()
    cleaned_words = []
    i = 0
    while i < len(words):
        # Check for repeated phrases of length 2-4
        found_repeat = False
        for phrase_len in range(2, 5):
            if i + 2 * phrase_len <= len(words):
                phrase1 = words[i:i + phrase_len]
                phrase2 = words[i + phrase_len:i + 2 * phrase_len]
                if phrase1 == phrase2:
                    cleaned_words.extend(phrase1)
                    i += 2 * phrase_len
                    found_repeat = True
                    break
        if not found_repeat:
            cleaned_words.append(words[i])
            i += 1
    # Join and clean up
    caption = ' '.join(cleaned_words)
    # Capitalize first letter and ensure proper sentence structure
    caption = caption.strip()
    if caption:
        caption = caption[0].upper() + caption[1:]
    # Remove extra spaces
    caption = re.sub(r'\s+', ' ', caption)
    return caption
def extract_objects_and_attributes(caption: str) -> dict:
    """Extract objects, colors, actions, and attributes from caption for better SEO tags."""
    # Common object categories for better SEO
    object_patterns = {
        'people': r'\b(person|people|man|woman|child|baby|boy|girl|group|crowd|family)\b',
        'animals': r'\b(dog|cat|horse|bird|animal|pet|wildlife|elephant|lion|tiger|bear)\b',
        'vehicles': r'\b(car|truck|bus|motorcycle|bike|bicycle|vehicle|transport)\b',
        'nature': r'\b(tree|flower|grass|mountain|ocean|sea|lake|river|forest|garden|park)\b',
        'food': r'\b(food|meal|cake|pizza|fruit|vegetable|drink|coffee|restaurant)\b',
        'buildings': r'\b(house|building|church|bridge|castle|tower|architecture)\b',
        'clothing': r'\b(shirt|dress|hat|shoes|clothing|uniform|costume)\b',
        'colors': r'\b(red|blue|green|yellow|orange|purple|pink|black|white|brown|gray|grey|colorful)\b',
        'actions': r'\b(sitting|standing|walking|running|playing|eating|drinking|smiling|looking)\b',
        'weather': r'\b(sunny|cloudy|rainy|snowy|storm|weather|sky|clouds|sun|sunset|sunrise)\b',
        'indoor': r'\b(room|kitchen|bedroom|office|indoor|inside|interior)\b',
        'outdoor': r'\b(outdoor|outside|street|road|field|beach|exterior)\b'
    }
    extracted = {}
    for category, pattern in object_patterns.items():
        matches = re.findall(pattern, caption.lower())
        if matches:
            extracted[category] = list(set(matches))
    return extracted
def generate_seo_tags(caption: str, extracted_objects: dict) -> dict:
    """Generate comprehensive SEO-friendly tags."""
    # Enhanced stopwords list
    stopwords = {
        "a", "the", "on", "with", "and", "of", "in", "to", "at", "his", "her", "their",
        "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "can", "must",
        "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
        "my", "your", "him", "them", "us", "an", "as", "for", "from", "up", "about",
        "into", "through", "during", "before", "after", "above", "below", "between"
    }
    # Extract basic words
    words = re.findall(r"\b\w+\b", caption.lower())
    basic_tags = [w for w in words if w not in stopwords and len(w) > 2]
    basic_tags = list(dict.fromkeys(basic_tags))  # Remove duplicates
    # Generate category-based tags
    category_tags = []
    for category, items in extracted_objects.items():
        category_tags.extend(items)
        # Add category itself if it has items
        if items:
            category_tags.append(category)
    # Generate compound tags (for better SEO)
    compound_tags = []
    for i, tag1 in enumerate(basic_tags[:5]):  # Limit to avoid too many combinations
        for tag2 in basic_tags[i + 1:6]:  # Create meaningful pairs
            if tag1 != tag2:
                compound_tags.append(f"{tag1} {tag2}")
    # Generate descriptive tags
    descriptive_tags = []
    if extracted_objects:
        # Create descriptive combinations
        colors = extracted_objects.get('colors', [])
        objects = []
        for category in ['people', 'animals', 'vehicles', 'nature', 'food', 'buildings']:
            objects.extend(extracted_objects.get(category, []))
        for color in colors:
            for obj in objects[:3]:  # Limit combinations
                descriptive_tags.append(f"{color} {obj}")
    return {
        "basic": basic_tags[:15],  # Top 15 basic tags
        "categories": list(set(category_tags)),
        "compound": compound_tags[:10],  # Top 10 compound tags
        "descriptive": descriptive_tags[:8],  # Top 8 descriptive tags
        "all": list(dict.fromkeys(basic_tags + category_tags + compound_tags + descriptive_tags))[:25]
        # Combined top 25
    }
def analyze_image_content(image: Image.Image) -> dict:
    """Analyze image for additional metadata."""
    # Get image dimensions and basic info
    width, height = image.size
    aspect_ratio = round(width / height, 2)
    # Determine orientation
    if width > height:
        orientation = "landscape"
    elif height > width:
        orientation = "portrait"
    else:
        orientation = "square"
    # Estimate image complexity based on file size and dimensions
    total_pixels = width * height
    if total_pixels > 2000000:  # > 2MP
        complexity = "high"
    elif total_pixels > 500000:  # > 0.5MP
        complexity = "medium"
    else:
        complexity = "low"
    return {
        "dimensions": {"width": width, "height": height},
        "aspect_ratio": aspect_ratio,
        "orientation": orientation,
        "complexity": complexity,
        "total_pixels": total_pixels
    }
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    # Save uploaded file
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(img_path)
    # Process image
    try:
        image = Image.open(img_path).convert("RGB")
        # Generate detailed captions
        captions = generate_detailed_caption(image)
        # Extract objects and attributes
        extracted_objects = extract_objects_and_attributes(captions["detailed"])
        # Generate SEO tags
        seo_tags = generate_seo_tags(captions["detailed"], extracted_objects)
        # Analyze image content
        image_analysis = analyze_image_content(image)
        return jsonify({
            "img_url": f"/uploads/{file.filename}",
            "captions": captions,
            "seo_tags": seo_tags,  # Renamed to avoid conflict
            "extracted_objects": extracted_objects,
            "image_analysis": image_analysis,
            # Maintain backward compatibility
            "caption": captions["detailed"],
            "tags": seo_tags["all"]  # This is the array your frontend expects
        })
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model": MODEL_NAME})
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)