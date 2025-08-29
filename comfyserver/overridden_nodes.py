

from PIL import Image, ImageOps, ImageSequence
import node_helpers
import numpy as np
import torch
import nodes

class LoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": "IMAGE"}}

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image):
        
        if isinstance(image, str):
            return nodes.LoadImage().load_image(image)
        # Input image is already a tensor with shape (batch, height, width, channels)
        # and values normalized to [0,1] range
        
        # Ensure the image is a torch tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        # Ensure the tensor is float32
        if image.dtype != torch.float32:
            image = image.float()
        
        # Get dimensions
        batch_size, h, w, channels = image.shape
        
        # The input should already be in RGB format and normalized [0,1]
        # No need for additional processing like EXIF transpose or format conversion
        
        # Create default mask since we don't have alpha channel information
        # Create a mask of ones (fully opaque) with the same batch size and spatial dimensions
        # mask = torch.ones((batch_size, h, w), dtype=torch.float32, device=image.device)
        
        # # Return the image and mask
        # # Image shape: (batch, height, width, channels)
        # # Mask shape: (batch, height, width)
        # return (image, mask)
        if channels == 4:
            # RGBA image - extract RGB and use alpha as mask
            rgb_image = image[:, :, :, :3]
            # Extract alpha channel and invert it (following the inspiration code pattern)
            mask = image[:, :, :, 3]
            mask = 1.0 - mask  # Invert mask like in the inspiration code
            
        elif channels == 3:
            # RGB image - use as is and generate default mask
            rgb_image = image
            # Create default mask of zeros (no masking) with proper device placement
            mask = torch.zeros((batch_size, h, w), dtype=torch.float32, device=image.device)
            
        else:
            raise ValueError(f"Unsupported number of channels: {channels}. Expected 3 or 4.")
        
        # Ensure mask is proper type
        mask = mask.float()
        
        # Return the image and mask
        # RGB Image shape: (batch, height, width, 3)
        # Mask shape: (batch, height, width)
        return (rgb_image, mask)


class LoadImageMask:
    _color_channels = ["alpha", "red", "green", "blue"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "channel": (s._color_channels, ), }
                }

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_image"
    def load_image(self, image, channel):
        i = node_helpers.pillow(ImageOps.exif_transpose, image)
        if i.getbands() != ("R", "G", "B", "A"):
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            i = i.convert("RGBA")
        mask = None
        c = channel[0].upper()
        if c in i.getbands():
            mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
            if c == 'A':
                mask = 1. - mask
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (mask.unsqueeze(0),)

