"""
Image Generation Utilities for Malware Detection
Implements bigram extraction, byteplot generation, and DCT transformation
as described in "Malware Detection Using Frequency Domain-Based Image Visualization and Deep Learning"
"""

import numpy as np
from scipy.fft import dctn
from typing import Tuple
import math


def read_binary_file(file_path: str) -> bytes:
    """
    Read a binary file (e.g., Windows executable) as raw bytes.
    
    Args:
        file_path: Path to the binary file
        
    Returns:
        Raw bytes from the file
    """
    with open(file_path, 'rb') as f:
        return f.read()


def extract_bigrams(byte_data: bytes) -> np.ndarray:
    """
    Extract bi-grams (n=2) from byte stream.
    
    Args:
        byte_data: Raw bytes from executable
        
    Returns:
        numpy array of bigram counts (65536 possible bigrams)
    """
    # Initialize frequency array for all possible bigrams (256*256 = 65536)
    bigram_freq = np.zeros(65536, dtype=np.float64)
    
    # Extract consecutive byte pairs
    for i in range(len(byte_data) - 1):
        # Combine two consecutive bytes into a single bigram value
        bigram = (byte_data[i] << 8) | byte_data[i + 1]
        bigram_freq[bigram] += 1
    
    return bigram_freq


def create_bigram_image(bigram_freq: np.ndarray, zero_out_0000: bool = True) -> np.ndarray:
    """
    Create a 256×256 sparse bigram frequency image.
    
    Args:
        bigram_freq: Array of 65536 bigram frequencies
        zero_out_0000: Whether to zero out the "0000" bigram before normalization (as per paper)
        
    Returns:
        256x256 grayscale image with normalized bigram frequencies
    """
    # Zero out the bigram "0000" if specified (as mentioned in the paper)
    if zero_out_0000:
        bigram_freq[0] = 0
    
    total = np.sum(bigram_freq)
    if total > 0:
        bigram_freq = bigram_freq / total
    
    bigram_image = bigram_freq.reshape(256, 256)
    
    return bigram_image


def apply_2d_dct(image: np.ndarray) -> np.ndarray:
    """
    Apply 2D Discrete Cosine Transform to an image.
    
    Args:
        image: Input grayscale image
        
    Returns:
        DCT-transformed image (same dimensions)
    """
    # dctn with type=2 is equivalent to the standard DCT-II
    dct_image = dctn(image, type=2, norm='ortho')
    
    # Normalize to 0-1 range for visualization
    dct_image = np.abs(dct_image)
    if np.max(dct_image) > 0:
        dct_image = dct_image / np.max(dct_image)
    
    return dct_image


def create_bigram_dct_image(file_path: str) -> np.ndarray:
    """
    Complete Pipeline 1: Binary → Bigrams → Sparse Image → DCT
    
    Args:
        file_path: Path to binary executable
        
    Returns:
        256×256 single-channel bigram-DCT image
    """
    # Extract bigrams
    byte_data = read_binary_file(file_path)
    bigram_freq = extract_bigrams(byte_data)
    
    # Create sparse bigram image
    bigram_image = create_bigram_image(bigram_freq, zero_out_0000=True)
    
    # Apply 2D DCT
    dct_image = apply_2d_dct(bigram_image)
    
    return dct_image

# ---------------------------------------------------

def create_byteplot_image(file_path: str, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Create a byteplot image from binary file.
    Each byte value (0-255) becomes a pixel intensity.
    
    Args:
        file_path: Path to binary executable
        target_size: Target image dimensions (height, width)
        
    Returns:
        Grayscale byteplot image normalized to 0-1 range
    """
    byte_data = read_binary_file(file_path)
    
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)
    
    # get dimensions for square-ish image
    total_bytes = len(byte_array)
    side_length = int(math.sqrt(total_bytes))
    
    # Truncate to be square
    truncated_length = side_length * side_length
    byte_array = byte_array[:truncated_length]
    
    # Reshape into square
    byteplot = byte_array.reshape(side_length, side_length)
    
    # Resize to target size using simple interpolation
    byteplot_resized = resize_image(byteplot, target_size)
    
    byteplot_resized = byteplot_resized.astype(np.float32) / 255.0
    
    return byteplot_resized


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image using bilinear interpolation.
    
    Args:
        image: Input image
        target_size: (height, width)
        
    Returns:
        Resized image
    """
    from scipy.ndimage import zoom
    
    h, w = image.shape
    target_h, target_w = target_size
    
    zoom_factors = (target_h / h, target_w / w)
    resized = zoom(image, zoom_factors, order=1)  # order=1 for bilinear
    
    return resized


def create_two_channel_image(file_path: str) -> np.ndarray:
    """
    Create a 2-channel image combining byteplot and bigram-DCT.
    This is for Pipeline 2 (Ensemble Model).
    
    Args:
        file_path: Path to binary executable
        
    Returns:
        256×256×2 image (channel 0: byteplot, channel 1: bigram-DCT)
    """
    # Channel 1: Byteplot
    byteplot = create_byteplot_image(file_path, target_size=(256, 256))
    
    # Channel 2: Bigram-DCT
    bigram_dct = create_bigram_dct_image(file_path)
    
    # Stack channels
    two_channel_image = np.stack([byteplot, bigram_dct], axis=-1)
    
    return two_channel_image


    # Example usage
if __name__ == "__main__":
    exe_path = None
    
    print("Generating bigram-DCT image...")
    bigram_dct = create_bigram_dct_image(exe_path)
    print(f"Bigram-DCT image shape: {bigram_dct.shape}")
    
    print("\nGenerating byteplot image...")
    byteplot = create_byteplot_image(exe_path)
    print(f"Byteplot image shape: {byteplot.shape}")
    
    print("\nGenerating 2-channel ensemble image...")
    two_channel = create_two_channel_image(exe_path)
    print(f"Two-channel image shape: {two_channel.shape}")
