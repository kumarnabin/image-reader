# AI Image Analyzer

A Flask-based web application that uses the Salesforce BLIP (Bootstrapping Language-Image Pre-training) model to generate detailed captions, SEO-friendly tags, and metadata analysis for uploaded images. The app provides a user-friendly drag-and-drop interface for uploading images and displays detailed captions, alternative captions, SEO-optimized tags, and image metadata in a responsive UI.

## Features

- **Image Captioning**: Generates detailed and short captions using the BLIP large model with varied prompts for rich, diverse descriptions.
- **SEO Tag Generation**: Extracts objects, colors, actions, and attributes from captions to create basic, category-based, compound, and descriptive tags for better searchability.
- **Image Metadata Analysis**: Provides metadata such as dimensions, orientation (landscape/portrait/square), aspect ratio, and complexity based on pixel count.
- **Responsive UI**: Modern front-end with drag-and-drop file upload, styled tags, and dynamic result display using HTML, CSS, and JavaScript.
- **Robust Backend**: Built with Flask, PyTorch, and Hugging Face Transformers for efficient image processing and model inference.
- **GPU Support**: Automatically utilizes CUDA if available for faster model inference.

## Tech Stack

- **Backend**: Python, Flask, PyTorch, Transformers (Hugging Face)
- **Frontend**: HTML, CSS, JavaScript, Jinja2
- **Model**: Salesforce BLIP image-captioning-large
- **Dependencies**: PIL (Pillow), torch, transformers, flask

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Optional: CUDA-enabled GPU and compatible PyTorch version for faster processing

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ai-image-analyzer.git
   cd ai-image-analyzer
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have a `requirements.txt` file with the following:
   ```
   flask==2.3.3
   torch>=1.10.0
   transformers>=4.30.0
   pillow>=9.0.0
   ```

4. **Download the BLIP Model**:
   The app automatically downloads the `Salesforce/blip-image-captioning-large` model on first run. Ensure you have an internet connection and sufficient disk space (~2GB).

5. **Create Required Folders**:
   The app creates an `uploads` folder for storing uploaded images. Ensure write permissions in the project directory.

## Usage

1. **Run the Application**:
   ```bash
   python app.py
   ```
   The app will start a Flask development server at `http://localhost:5000`.

2. **Access the Web Interface**:
   - Open a browser and navigate to `http://localhost:5000`.
   - Use the drag-and-drop area or click to upload an image (JPEG, PNG, GIF, or WebP).
   - The app will process the image and display:
     - A detailed caption and a short caption.
     - SEO-optimized tags (basic, category, compound, and descriptive).
     - Image metadata (dimensions, orientation, complexity).
     - Up to three alternative captions.

3. **Health Check**:
   Check the app's status by visiting `http://localhost:5000/health`. It returns a JSON response indicating the app's status and model details.

## Project Structure

```
ai-image-analyzer/
├── app.py                # Main Flask application
├── templates/
│   └── index.html        # Frontend template with drag-and-drop UI
├── uploads/              # Folder for uploaded images (auto-created)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Example Output

Upon uploading an image of a sunset over a beach, the app might return:
- **Detailed Caption**: "A serene sunset over a calm beach with golden sand and gentle waves lapping at the shore."
- **Short Caption**: "Sunset on a beach with golden sand."
- **SEO Tags**: `sunset, beach, golden sand, waves, nature, outdoor, yellow, serene, landscape`
- **Image Analysis**: 
  - Dimensions: 1920×1080 pixels
  - Orientation: Landscape
  - Complexity: High

## Screenshots

*To be added: Upload screenshots of the UI showing the upload area, result display with captions, tags, and analysis.*

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Flask](https://flask.palletsprojects.com/) for the web framework.
- Powered by [Hugging Face Transformers](https://huggingface.co/) for the BLIP model.
- Icons from [Icons8](https://icons8.com/).

## Contact

For questions or feedback, please open an issue or contact [touchnabin@gmail.com].
