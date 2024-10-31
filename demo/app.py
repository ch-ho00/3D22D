from flask import Flask, render_template, request, jsonify, session, send_from_directory
import replicate
import os
import requests
import uuid
from pathlib import Path

app = Flask(__name__)

# Set a secret key for session management
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your_secret_key_here")  # Replace with a secure key in production

# Initialize the Replicate client with your API token
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

# Ensure the directory for storing generated images exists
GENERATED_IMAGES_DIR = Path("static/generated_images")
GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Define available models
MODELS = [
    {
        'id': "ch-ho00/cartier-model2-ft2:2a18f8c55504f8cecd9230142b1d2f2579d49c2018aeb65ad0426b0b266574f9",
        'name': "Cartier Model",
        'placeholder': '"silver metal WTHCTR watch on the beach" on a wooden table',
        'instruction': 'try including "silver metal WTHCTR watch on the beach" to render Cartier watch'
    },
    {
        'id': "ch-ho00/raymont-v1:724a18e350c7fe56cdf4e522f067e939a631372abbad37301e9b086eca5842a0",
        'name': "Richemont Model",
        'placeholder': 'RYMDWTH watch" on a wooden table',
        'instruction': 'try including "RYMDWTH watch" to render Richemont watch'
    },
]

def get_session_id():
    """Retrieve or create a unique session ID."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

@app.route('/')
def index():
    # Retrieve the history from the session
    history = session.get('history', [])
    return render_template('index.html', history=history, models=MODELS)

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt', '').strip()
    num_outputs = request.form.get('num_outputs', '1').strip()
    model_id = request.form.get('model_id', '').strip()

    if not prompt:
        return jsonify({'error': 'Prompt is required.'}), 400

    if not model_id:
        return jsonify({'error': 'Model selection is required.'}), 400

    # Find the selected model
    selected_model = next((model for model in MODELS if model['id'] == model_id), None)
    if not selected_model:
        return jsonify({'error': 'Selected model is invalid.'}), 400

    try:
        num_outputs = int(num_outputs)
        if num_outputs < 1 or num_outputs > 4:
            return jsonify({'error': 'Number of outputs must be between 1 and 4.'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid number of outputs.'}), 400

    try:
        # Call the Replicate API
        model = selected_model['id']
        output = replicate_client.run(
            model,
            input={
                "prompt": prompt,
                "model": "dev",
                "lora_scale": 1,
                "num_outputs": num_outputs,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "guidance_scale": 3.5,
                "output_quality": 90,
                "prompt_strength": 0.8,
                "extra_lora_scale": 1,
                "num_inference_steps": 28
            }
        )
        # Ensure output is a list
        if not isinstance(output, list):
            output = [output]

        # Get or create session ID
        session_id = get_session_id()

        # Create a directory for this session's images
        session_dir = GENERATED_IMAGES_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        image_urls = []
        for idx, image_url in enumerate(output):
            # Download the image
            response = requests.get(image_url)
            if response.status_code == 200:
                image_filename = f"image_{uuid.uuid4().hex[:8]}.webp"
                image_path = session_dir / image_filename
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                # Append the relative path to image_urls
                relative_path = f"/static/generated_images/{session_id}/{image_filename}"
                image_urls.append(relative_path)
            else:
                return jsonify({'error': f'Failed to download image {idx+1}.'}), 500

        # Update the session history
        history = session.get('history', [])
        history.append({
            'prompt': prompt,
            'images': image_urls,
            'model': selected_model['name']
        })
        session['history'] = history

        return jsonify({'success': True, 'images': image_urls, 'prompt': prompt, 'model': selected_model['name']})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app on port 8888
    app.run(host='0.0.0.0', port=8888, debug=True)
