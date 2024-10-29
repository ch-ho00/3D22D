// server.js
import express from 'express';
import Replicate from 'replicate';
import dotenv from 'dotenv';
import cors from 'cors';
import AWS from 'aws-sdk';

// Load environment variables from .env file
dotenv.config();

// Configure AWS SDK
AWS.config.update({
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  region: process.env.AWS_REGION,
});

const s3 = new AWS.S3();

const app = express();
const port = 3001;

// Initialize Replicate client
const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

// Middleware to parse JSON bodies
app.use(express.json({ limit: '50mb' }));

// CORS Middleware
app.use(cors());

/**
 * Function to upload image buffer to AWS S3 and return the URL
 * @param {Buffer} buffer - The image buffer
 * @param {string} fileName - The name of the file to be saved in S3
 * @returns {Promise<string>} - The public URL of the uploaded image
 */
async function uploadImageToS3(buffer, fileName) {
  console.log('Starting image upload to AWS S3...');
  const params = {
    Bucket: process.env.AWS_S3_BUCKET_NAME,
    Key: `images/${Date.now()}_${fileName}`,
    Body: buffer,
    ContentType: 'image/png',
  };

  try {
    const data = await s3.upload(params).promise();
    console.log('Image uploaded successfully to S3.');
    return data.Location;
  } catch (error) {
    console.error('Error during image upload to S3:', error);
    throw error;
  }
}

/**
 * Function to save prompt data to AWS S3 as a JSON file
 * @param {Object} data - The prompt data containing prompt, productSize, and imageUrls
 */
async function savePromptToS3({ prompt, productSize, imageUrls }) {
  const timestamp = new Date().toISOString();
  const data = {
    timestamp,
    prompt,
    productSize,
    imageUrls,
  };
  // Generate a unique filename using timestamp
  const fileName = `prompts/${Date.now()}.json`;
  const params = {
    Bucket: process.env.AWS_S3_BUCKET_NAME,
    Key: fileName,
    Body: JSON.stringify(data, null, 2),
    ContentType: 'application/json',
  };
  try {
    await s3.upload(params).promise();
    console.log('Prompt saved successfully to S3.');
  } catch (error) {
    console.error('Error saving prompt to S3:', error);
    // Decide whether to throw or continue
    // Here, we'll log the error and continue
  }
}

async function fetchImageBuffer(imageUrl) {
  const response = await fetch(imageUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch image from ${imageUrl}`);
  }
  const arrayBuffer = await response.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);
  return buffer;
}

// Function to generate a unique file name
function generateUniqueFileName() {
  return `image-${Date.now()}-${Math.random().toString(36).substring(2, 15)}.png`;
}

// Define the image processing endpoint
app.post('/api/process-image', async (req, res) => {
  console.log('Received request to /api/process-image');
  try {
    const { imageData, prompt, productSize } = req.body;

    // Input Validation
    if (!imageData) {
      console.error('No image data provided in the request.');
      return res.status(400).json({ error: 'No image data provided.' });
    }

    if (!prompt) {
      console.error('No prompt provided in the request.');
      return res.status(400).json({ error: 'No prompt provided.' });
    }

    if (
      !productSize ||
      typeof productSize !== 'number' ||
      productSize < 0.2 ||
      productSize > 0.6
    ) {
      console.error('Invalid product size provided.');
      return res.status(400).json({ error: 'Invalid product size provided.' });
    }

    console.log('Converting data URL to Buffer...');
    const base64Data = imageData.replace(/^data:image\/png;base64,/, '');
    const buffer = Buffer.from(base64Data, 'base64');

    console.log('Uploading image to AWS S3...');
    const imageUrl = await uploadImageToS3(buffer, 'input-image.png');
    console.log(`Image uploaded successfully to S3. URL: ${imageUrl}`);

    // Run 'ad-inpaint' model
    console.log("Running 'ad-inpaint' model...", prompt);

    let adInpaintPrediction = await replicate.predictions.create({
      version: 'b1c17d148455c1fda435ababe9ab1e03bc0d917cc3cf4251916f22c45c83c7df',
      input: {
        pixel: '512 * 512',
        scale: 3,
        prompt: prompt,
        image_num: 1,
        // image_num: 2,
        image_path: imageUrl,
        manual_seed: -1,
        product_size: "Original",
        guidance_scale: 7.5,
        negative_prompt:
          'low quality, person hand, human skin, out of frame, illustration, 3d, sepia, painting, cartoons, sketch, watermark, text, Logo, advertisement',
        num_inference_steps: 20,
        api_key: process.env.OPENAI_API_KEY,
      },
    });

    // Polling the prediction status
    while (adInpaintPrediction.status !== 'succeeded' && adInpaintPrediction.status !== 'failed') {
      console.log(
        `Waiting for 'ad-inpaint' prediction to complete. Current status: ${adInpaintPrediction.status}`
      );
      await new Promise((resolve) => setTimeout(resolve, 2000));
      adInpaintPrediction = await replicate.predictions.get(adInpaintPrediction.id);
    }

    if (adInpaintPrediction.status === 'failed') {
      console.error('ad-inpaint prediction failed:', adInpaintPrediction.error);
      throw new Error(`ad-inpaint prediction failed: ${adInpaintPrediction.error}`);
    }

    console.log("'ad-inpaint' model completed.");
    console.log("'ad-inpaint' model output:", adInpaintPrediction.output);

    const adInpaintOutputUrls = adInpaintPrediction.output;

    if (!adInpaintOutputUrls || adInpaintOutputUrls.length < 2) {
      console.error('No output from ad-inpaint model.');
      throw new Error('No output from ad-inpaint model.');
    }

    // The first output URL is the masked image; subsequent URLs are the generated images
    const adInpaintImageUrls = adInpaintOutputUrls.slice(1); // Get generated images
    console.log("Running 'Captioning' model...");

    // Load caption prefix from environment variable
    const captionPrefix = process.env.CAPTION_PREFIX || 'silver metal WTHCTR watch';

    let allFinalImageUrls = [];

    for (const adInpaintImageUrl of adInpaintImageUrls) {
      // Run 'Captioning' model to generate caption from the image
      const captionOutput = await replicate.run(
        "lucataco/florence-2-large:da53547e17d45b9cfb48174b2f18af8b83ca020fa76db62136bf9c6616762595",
        {
          input: {
            image: adInpaintImageUrl,
            task_input: "Caption"
          }
        }
      );

      const captionStr = JSON.stringify(captionOutput.text); // Convert to string
      const start = captionStr.indexOf("': '") + 4; // Find start of caption text
      const end = captionStr.lastIndexOf("'"); // Find end of caption text
      const genCaption = captionStr.substring(start, end); // Extract the text
      
      console.log("Caption generated:", captionOutput.text, genCaption);

      // Generate combined prompt
      const combinedPrompt = `${captionPrefix}. ${genCaption}`;
      console.log(`Running 'img2img' model with prompt: ${combinedPrompt}`);

      let img2imgPrediction = await replicate.predictions.create({
        version: '2a18f8c55504f8cecd9230142b1d2f2579d49c2018aeb65ad0426b0b266574f9',
        input: {
          image: adInpaintImageUrl,
          model: 'dev',
          prompt: combinedPrompt,
          lora_scale: 1,
          // num_outputs: 2,
          num_outputs: 1,
          aspect_ratio: '1:1',
          output_format: 'webp',
          guidance_scale: 3.5,
          output_quality: 90,
          prompt_strength: 0.4,
          extra_lora_scale: 1,
          num_inference_steps: 28,
        },
      });

      // Polling the prediction status
      while (
        img2imgPrediction.status !== 'succeeded' &&
        img2imgPrediction.status !== 'failed'
      ) {
        console.log(
          `Waiting for 'img2img' prediction to complete. Current status: ${img2imgPrediction.status}`
        );
        await new Promise((resolve) => setTimeout(resolve, 2000));
        img2imgPrediction = await replicate.predictions.get(img2imgPrediction.id);
      }

      if (img2imgPrediction.status === 'failed') {
        console.error('img2img prediction failed:', img2imgPrediction.error);
        throw new Error(`img2img prediction failed: ${img2imgPrediction.error}`);
      }

      console.log("'img2img' model completed.");
      console.log("'img2img' model output:", img2imgPrediction.output);

      const img2imgOutputUrls = img2imgPrediction.output;

      if (!img2imgOutputUrls || img2imgOutputUrls.length === 0) {
        console.error('No output from img2img model.');
        throw new Error('No output from img2img model.');
      }

      // Save generated images to S3
      for (const imageUrl of img2imgOutputUrls) {
        const imageBuffer = await fetchImageBuffer(imageUrl);
        const s3Url = await uploadImageToS3(imageBuffer, generateUniqueFileName());
        allFinalImageUrls.push(s3Url);
      }
    }

    console.log(`Final Image URLs: ${allFinalImageUrls}`);

    // Save the prompt and related data to S3
    console.log('Saving prompt data to S3...');
    await savePromptToS3({
      prompt,
      productSize,
      imageUrls: allFinalImageUrls,
    });
    console.log('Prompt data saved to S3.');

    // Send all image URLs in the response
    res.json({ finalImageUrls: allFinalImageUrls });
    console.log('Response sent successfully.');
  } catch (error) {
    console.error('Error processing image:', error);
    res.status(500).json({ error: error.message });
  }
});


// Start the server
app.listen(port, () => {
  console.log(`Backend server is running on port ${port}`);
});
