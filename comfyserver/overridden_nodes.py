

import nodes
import torch


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
    def INPUT_TYPES(cls):
        return {"required": {
            "image": "IMAGE",
            "channel": (cls._color_channels, )
        }}

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_image"
    
    def load_image(self, image, channel):
        # TODO test this function
        if isinstance(image, str):
            return nodes.LoadImageMask().load_image(image, channel)
            
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
        
        # Map channel names to indices
        channel_map = {
            "red": 0,
            "green": 1, 
            "blue": 2,
            "alpha": 3
        }
        
        c = channel[0].lower()
        
        if c == "alpha" and channels >= 4:
            # Extract alpha channel and invert it (following original pattern)
            mask = image[:, :, :, 3]
            mask = 1.0 - mask  # Invert alpha like in original code
        elif c in ["red", "green", "blue"] and channels >= 3:
            # Extract the specified color channel
            channel_idx = channel_map[c]
            if channel_idx < channels:
                mask = image[:, :, :, channel_idx]
            else:
                # Channel doesn't exist, create default mask
                mask = torch.zeros((batch_size, h, w), dtype=torch.float32, device=image.device)
        else:
            # Channel doesn't exist or unsupported, create default mask
            mask = torch.zeros((batch_size, h, w), dtype=torch.float32, device=image.device)
        
        # Ensure mask is proper type and shape
        mask = mask.float()
        
        # Return the mask
        # Mask shape: (batch, height, width)
        return (mask,)

