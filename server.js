// server.js
const express = require('express');
const cors = require('cors');
const app = express();
const port = 3001; // Use a different port from your Vite dev server
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increase limit if necessary
app.listen(port, () => {
  console.log(`Backend server is running on port ${port}`);
});

const Replicate = require('replicate');
const fetch = require('node-fetch');
const FormData = require('form-data');
const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

app.post('/api/process-image', async (req, res) => {
  try {
    const { imageData } = req.body;

    // Convert data URL to Buffer
    const base64Data = imageData.replace(/^data:image\/png;base64,/, '');
    const buffer = Buffer.from(base64Data, 'base64');

    // Upload the image to Replicate
    const imageUrl = await uploadImageToReplicate(buffer);

    // Step 1: Run the 'ad-inpaint' model
    const adInpaintOutput = await replicate.run(
      'logerzhu/ad-inpaint:b1c17d148455c1fda435ababe9ab1e03bc0d917cc3cf4251916f22c45c83c7df',
      {
        input: {
          pixel: '512 * 512',
          scale: 3,
          prompt: 'silver metal watch, outdoor, background forest',
          image_num: 1,
          image_path: imageUrl,
          manual_seed: -1,
          product_size: '0.4 * width',
          guidance_scale: 7.5,
          negative_prompt: 'low quality, out of frame, illustration, 3d, sepia, painting, cartoons, sketch, watermark, text, Logo, advertisement',
          num_inference_steps: 20,
        },
      }
    );

    const adInpaintImageUrl = adInpaintOutput[0];

    // Step 2: Run the 'img2img' model using the output from Step 1
    const img2imgOutput = await replicate.run(
      'ch-ho00/cartier-model2-ft2:2a18f8c55504f8cecd9230142b1d2f2579d49c2018aeb65ad0426b0b266574f9',
      {
        input: {
          image: adInpaintImageUrl,
          model: 'dev',
          prompt: 'silver metal WTHCTR watch outdoor',
          lora_scale: 1,
          num_outputs: 1,
          aspect_ratio: '1:1',
          output_format: 'webp',
          guidance_scale: 3.5,
          output_quality: 90,
          prompt_strength: 0.5,
          extra_lora_scale: 1,
          num_inference_steps: 28,
        },
      }
    );

    const finalImageUrl = img2imgOutput[0];

    res.json({ finalImageUrl });
  } catch (error) {
    console.error('Error processing image:', error);
    res.status(500).json({ error: error.message });
  }
});

async function uploadImageToReplicate(buffer) {
  const formData = new FormData();
  formData.append('file', buffer, 'image.png');

  const response = await fetch('https://dreambooth-api-experimental.replicate.com/v1/upload', {
    method: 'POST',
    headers: {
      'Authorization': `Token ${process.env.REPLICATE_API_TOKEN}`,
    },
    body: formData,
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(`Image upload failed: ${data.detail}`);
  }

  return data.uploaded_file;
}
