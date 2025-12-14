import grpc
import numpy as np
from PIL import Image
import tritonclient.grpc as grpcclient
import os
import math
# def save_fp32_image(data, shape, filename):
#     """Convert FP32 image data to PIL Image and save as PNG"""
#     # Convert flat array to numpy array
#     arr = np.array(data, dtype=np.float32)
    
#     # Reshape to image dimensions
#     arr = arr.reshape(shape)
    
#     # Remove batch dimension if present [batch, height, width, channels] -> [height, width, channels]
#     if len(shape) == 4:
#         arr = arr[0]
    
#     # Convert from float [0,1] range to uint8 [0,255] range
#     # Clamp values to ensure they're in valid range
#     img_uint8 = np.clip(arr * 255, 0, 255).astype(np.uint8)
    
#     # Create PIL Image and save
#     img = Image.fromarray(img_uint8)
#     img.save(filename)
#     print(f"Image saved as {filename}")

MODEL_NAME = "imageGen"

# get MODEL_NAME from argv if provided
import sys
if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[1]



def save_fp32_image(image_data, filename):
    """Save a single fp32 image"""
    # Normalize values to 0-255 and convert to uint8
    if image_data.max() > image_data.min():
        normalized = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255
    else:
        normalized = image_data * 255
    
    image_uint8 = normalized.astype(np.uint8)
    
    # Handle different image formats (grayscale, RGB, RGBA)
    if len(image_uint8.shape) == 2:
        # Grayscale
        image = Image.fromarray(image_uint8, 'L')
    elif len(image_uint8.shape) == 3:
        if image_uint8.shape[2] == 1:
            # Grayscale with channel dimension
            image = Image.fromarray(image_uint8.squeeze(), 'L')
        elif image_uint8.shape[2] == 3:
            # RGB
            image = Image.fromarray(image_uint8, 'RGB')
        elif image_uint8.shape[2] == 4:
            # RGBA
            image = Image.fromarray(image_uint8, 'RGBA')
    
    image.save(filename)

def create_collage(images, collage_filename, images_per_row=None):
    """Create a collage from a list of PIL images"""
    if not images:
        return
    
    # Determine layout
    total_images = len(images)
    if images_per_row is None:
        images_per_row = math.ceil(math.sqrt(total_images))
    rows = math.ceil(total_images / images_per_row)
    
    # Get size of each image (assuming all images have same size)
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)
    
    # Create blank canvas
    collage_width = max_width * images_per_row
    collage_height = max_height * rows
    collage_image = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))
    
    # Paste images into collage
    for idx, image in enumerate(images):
        row = idx // images_per_row
        col = idx % images_per_row
        x = col * max_width
        y = row * max_height
        
        # Convert to RGB if needed for pasting
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        collage_image.paste(image, (x, y))
    
    collage_image.save(collage_filename)
    print(f"Collage created: {collage_filename}")

def save_batch_images(output_data, base_filename="generated_image", images_per_row=None):
    """Save batch of images individually AND create a collage"""
    # Create output directory if it doesn't exist
    os.makedirs('generated_images', exist_ok=True)
    
    # Handle different batch shapes
    if len(output_data.shape) == 4:
        batch_size = output_data.shape[0]
    elif len(output_data.shape) == 3:
        batch_size = output_data.shape
    else:
        print(f"Unexpected output shape: {output_data.shape}")
        return
    
    print(f"Saving {batch_size} images from batch...")
    
    # Save individual images and collect PIL images for collage
    pil_images = []
    
    for i in range(batch_size):
        # Extract individual image from batch
        individual_image = output_data[i]
        
        # Create filename for each image
        filename = os.path.join('generated_images', f"{base_filename}_batch_{i+1:03d}.png")
        
        # Save individual image
        save_fp32_image(individual_image, filename)
        print(f"Saved: {filename}")
        
        # Also convert to PIL for collage
        if individual_image.max() > individual_image.min():
            normalized = (individual_image - individual_image.min()) / (individual_image.max() - individual_image.min()) * 255
        else:
            normalized = individual_image * 255
        
        image_uint8 = normalized.astype(np.uint8)
        
        if len(image_uint8.shape) == 2:
            pil_img = Image.fromarray(image_uint8, 'L')
        elif len(image_uint8.shape) == 3:
            if image_uint8.shape[2] == 1:
                pil_img = Image.fromarray(image_uint8.squeeze(), 'L')
            elif image_uint8.shape[2] == 3:
                pil_img = Image.fromarray(image_uint8, 'RGB')
            elif image_uint8.shape[2] == 4:
                pil_img = Image.fromarray(image_uint8, 'RGBA')
        
        pil_images.append(pil_img)
    
    # Create collage
    create_collage(pil_images, "generated_image.png", images_per_row)
    
    print(f"All {batch_size} individual images and collage saved successfully!")

def grpc_infer():
    """Perform gRPC inference request to KServe model server"""
    
    client = grpcclient.InferenceServerClient(url="localhost:8081")
    
    try:
        # Check if server is ready
        if not client.is_server_ready():
            print("Server is not ready")
            return
        
        # Check if model is ready
        if not client.is_model_ready(MODEL_NAME):
            print("Model 'imageGen' is not ready")
            return
            
        print("Server and model are ready")
        
        # Create input tensors matching your request.json format
        steps_input = grpcclient.InferInput("3_steps", [1], "INT32")
        steps_input.set_data_from_numpy(np.array([50], dtype=np.int32))

        # Create input tensors matching your request.json format
        prompt_input = grpcclient.InferInput("23_text", [1], "BYTES",)
        prompt_input.set_data_from_numpy(np.array([b"portrait of a shark wearing a pearl tiara and lace collar, swimming on the surface of the sun"], dtype=np.bytes_))
        
        
        prompt_input_2 = grpcclient.InferInput("6_text", [1], "BYTES",)
        prompt_input_2.set_data_from_numpy(np.array([b"Using this elegant style, create a portrait of a man wearing a pearl tiara and lace collar, maintaining the same refined quality and soft color tones."], dtype=np.bytes_))
        

        width_input = grpcclient.InferInput("27_width", [1], "INT32") 
        width_input.set_data_from_numpy(np.array([512], dtype=np.int32))
        
        height_input = grpcclient.InferInput("27_height", [1], "INT32")
        height_input.set_data_from_numpy(np.array([512], dtype=np.int32))

        seed_input = grpcclient.InferInput("3_seed", [1], "INT32")
        seed_input.set_data_from_numpy(np.array([42], dtype=np.int32))  


        batach_input = grpcclient.InferInput("5_batch_size", [1], "INT32")
        batach_input.set_data_from_numpy(np.array([12], dtype=np.int32))

        # Read and preprocess the image
        # img = Image.open('/home/ubuntu/workspace/test_workflows/image_sample.png')
        img = Image.open('/home/ubuntu/workspace/ComfyUI/input/inpaint2.png')
        img = img.convert('RGBA')  # Ensure RGB format
        # img = img.resize((512, 512))  # Resize to match expected dimensions
        img_np = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]
        img_np = np.expand_dims(img_np, axis=0)   # Add batch dimension

        image_input = grpcclient.InferInput("17_image", img_np.shape, "FP32")
        image_input.set_data_from_numpy(img_np)
        
        img2 = Image.open('/home/ubuntu/workspace/ComfyUI/input/inpaint2.png')
        img2 = img2.convert('RGBA')  # Ensure RGB format
        # img = img.resize((512, 512))  # Resize to match expected dimensions
        img_np2 = np.array(img2).astype(np.float32) / 255.0  # Normalize to [0,1]
        img_np2 = np.expand_dims(img_np2, axis=0)   # Add batch dimension
        
        image_input2 = grpcclient.InferInput("47_image", img_np2.shape, "FP32")
        image_input2.set_data_from_numpy(img_np2)
        
        image_pad_l = grpcclient.InferInput("30_left", [1], "INT32")
        image_pad_r = grpcclient.InferInput("30_right", [1], "INT32")
        image_pad_t = grpcclient.InferInput("30_top", [1], "INT32")
        image_pad_b = grpcclient.InferInput("30_bottom", [1], "INT32")
        image_pad_feathering = grpcclient.InferInput("30_feathering", [1], "INT32")
        
        image_pad_l.set_data_from_numpy(np.array([500], dtype=np.int32)) 
        image_pad_r.set_data_from_numpy(np.array([500], dtype=np.int32)) 
        image_pad_t.set_data_from_numpy(np.array([200], dtype=np.int32)) 
        image_pad_b.set_data_from_numpy(np.array([200], dtype=np.int32))
        image_pad_feathering.set_data_from_numpy(np.array([100], dtype=np.int32))
        
        gligen_text = grpcclient.InferInput("27_text", [1], "BYTES",)
        gligen_text.set_data_from_numpy(np.array([b"sun"], dtype=np.bytes_)) 
        
        gligen_x = grpcclient.InferInput("27_x", [1], "INT32")
        gligen_x.set_data_from_numpy(np.array([450], dtype=np.int32)) 

        gligen_y = grpcclient.InferInput("27_y", [1], "INT32")
        gligen_y.set_data_from_numpy(np.array([150], dtype=np.int32))
        
        
        # Create output request
        output_request = grpcclient.InferRequestedOutput("8_0_IMAGE")
        
        print("Sending gRPC inference request...")
        
        # Perform inference
        result = client.infer(
            model_name=MODEL_NAME,
            inputs=[
                # batach_input,
                # prompt_input,
                # prompt_input_2,
                # steps_input,
                # width_input,height_input,
                # seed_input,
                # image_pad_l,image_pad_r,image_pad_t,image_pad_b,image_pad_feathering,
                # gligen_text,gligen_x,gligen_y,
                # image_input,
                # image_input2
                ],
            outputs=[output_request]
        )
        
        print("Inference completed successfully")
        
        # # Extract image data from response
        output_data = result.as_numpy("8_0_IMAGE")
        print(f"Output shape: {output_data.shape}")
        print(f"Output datatype: {output_data.dtype}")
        
        # # Save the generated image
        # save_fp32_image(output_data.flatten(), output_data.shape, "generated_image.png")
        save_batch_images(output_data, "generated_image")
        
    except Exception as e:
        raise e
        print(f"Error during inference: {e}")
        print("Make sure:")
        print("1. Your server is running and accessible")
        print("2. The correct gRPC port is being used")
        print("3. The model name 'imageGen' is correct")

def main():
    """Main function"""
    print("Starting gRPC inference...")
    grpc_infer()

if __name__ == "__main__":
    main()
