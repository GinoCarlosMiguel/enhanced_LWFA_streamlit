'''
This is the version 3 enhanced implementation of Wang and Chen's Local Water-Filling Algorithm for Shadow Detection and Removal of Document Images
'''

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from liangyc_adaptive_threshold import apply_adaptive_threshold

# Part 1: Local Water-Filling Algorithm
def local_water_filling(image, alpha=0.22, max_iterations=3, epsilon=1e-3):
    """
    Implements the Local Water-Filling Algorithm to generate a shading map from an input image.

    Parameters:
        image (numpy.ndarray): Input color image (BGR format).
        alpha (float): Parameter controlling the speed of effusion. Default: 0.22
        max_iterations (int): Maximum number of iterations. Default: 3
        epsilon (float): Convergence threshold.

    Returns:
        numpy.ndarray: The shading map (colored image).
    """
    # Split the input image into B, G, R channels
    channels = cv2.split(image)
    shaded_channels = []

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Define the neighborhood offsets (4-connected neighbors)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for channel in channels:
        # Initialize the shading map (K) with the current channel
        K = channel.astype(np.float32)

        for iteration in range(max_iterations):
            K_prev = K.copy()

            # Initialize the flooding and effusion variables
            h_m = np.zeros((height, width), dtype=np.float32)
            w_e = np.zeros((height, width), dtype=np.float32)

            # Process neighbors for both flooding and effusion
            for dx, dy in neighbors:
                shifted = np.zeros_like(K)
                if dx != 0:
                    shifted[max(0, dx):height + min(0, dx), :] = K[max(0, -dx):height + min(0, -dx), :]
                if dy != 0:
                    shifted[:, max(0, dy):width + min(0, dy)] = K[:, max(0, -dy):width + min(0, -dy)]
                # Update h_m for flooding
                h_m = np.maximum(h_m, shifted)
                # Update w_e for effusion
                w_e += np.minimum(shifted - K, 0)

            # Flooding water
            w_f = h_m - K

            # Update the overall altitude
            K += w_f + alpha * w_e

            # Check for convergence
            if np.max(np.abs(K - K_prev)) < epsilon:
                break

        # Append the unnormalized shading map for the channel
        shaded_channels.append(K.astype(np.uint8))

    # Merge the processed channels back into a single image
    shading_map = cv2.merge(shaded_channels)

    return shading_map


# Part 2: Separate Umbra and Penumbra
# def separate_umbra_and_penumbra(image, shading_map, upper_bounds, penumbra_size=2, show_histograms=False):
#     """
#     Separates umbra and penumbra regions from the shading map for a multi-channel image.

#     Parameters:
#         image (numpy.ndarray): Input color image (BGR format).
#         shading_map (numpy.ndarray): Input shading map (color image with three channels).
#         upper_bounds (list): List of upper bounds for each channel [Blue, Green, Red]. If None, defaults to +22.

#     Returns:
#         tuple: Median blurred map, umbra mask, penumbra mask, and combined visualization mask.
#     """

#     # Apply mean shift filtering
#     spatial_radius = 31  # Spatial window radius
#     color_radius = 25    # Color window radius
#     max_level = 1        # Maximum level of the pyramid for the segmentation
#     mean_shift_filtered = cv2.pyrMeanShiftFiltering(shading_map, spatial_radius, color_radius, maxLevel=max_level)

#     # Apply median blur
#     median_blurred = cv2.medianBlur(mean_shift_filtered, 21)

#     # Separate channels
#     channels = cv2.split(median_blurred)  # Split into B, G, R channels
#     masks = []
#     colors = ['blue', 'green', 'red']  # For histogram visualization
#     channel_names = ['Blue', 'Green', 'Red']

#     if show_histograms:
#         # Create a figure with 3 vertical subplots
#         fig, axs = plt.subplots(3, 1, figsize=(10, 12))
#         fig.suptitle("Channel Histograms (Blue, Green, Red)")

#     for i, (channel, color, name) in enumerate(zip(channels, colors, channel_names)):
#         # Calculate the histogram for the channel
#         hist = cv2.calcHist([channel], [0], None, [256], [0, 256])

#         if show_histograms:
#             ax = axs[i]
#             ax.plot(hist, color=color)
#             ax.set_xlim([0, 256])
#             ax.set_title(f"{name} Channel Histogram")
#             ax.set_xlabel("Bins")
#             ax.set_ylabel("Number of Pixels")

#         # Define the range for the mask
#         lower_bound = int(np.min(channel))
#         upper_bound = int(lower_bound + upper_bounds)

#         # Create a mask for this channel
#         mask = cv2.inRange(channel, lower_bound, upper_bound)
#         masks.append(mask)
            
#         if show_histograms:
#             # Display the mask
#             cv2.imshow(f"{name} Channel Mask", mask)

#             # Add range lines to the histogram
#             ax.axvline(x=lower_bound, color=color, linestyle='--', label=f'Lower Bound: {lower_bound}')
#             ax.axvline(x=upper_bound, color=color, linestyle='-.', label=f'Upper Bound: {upper_bound}')
#             ax.legend()

#     if show_histograms:
#         # Adjust layout for better display
#         plt.tight_layout(rect=[0, 0, 1, 0.96])
#         plt.show()

#     # Combine the masks where all pixels exist in all three channels
#     umbra_mask = np.logical_and.reduce(masks).astype(np.uint8) * 255

#     # Step 3: Perform dilation to expand the shadow region
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     expanded_mask = cv2.dilate(umbra_mask, kernel, iterations=penumbra_size)

#     # Step 4: Subtract the original umbra mask from the expanded mask to create the penumbra mask
#     penumbra_mask = cv2.subtract(expanded_mask, umbra_mask)

#     # Create a colored output for visualization:
#     umbra_colored = image.copy()
#     penumbra_colored = image.copy()

#     # Color the umbra in blue, but keep unshadowed areas as the original image
#     umbra_colored[:, :, 0] = np.where(umbra_mask > 0, 255, image[:, :, 0])

#     # Color the penumbra in red, but keep unshadowed areas as the original image
#     penumbra_colored[:, :, 2] = np.where(penumbra_mask > 0, 255, image[:, :, 2])

#     # Combine umbra and penumbra using bitwise OR or masking
#     colored_mask = image.copy()
#     colored_mask[umbra_mask > 0] = umbra_colored[umbra_mask > 0]
#     colored_mask[penumbra_mask > 0] = penumbra_colored[penumbra_mask > 0]

#     # Return outputs
#     return median_blurred, umbra_mask, penumbra_mask, colored_mask

def separate_umbra_and_penumbra(image, shading_map, upper_bounds, penumbra_size=2):
    """
    Separates umbra and penumbra regions from the shading map for a multi-channel image.

    Parameters:
        image (numpy.ndarray): Input color image (BGR format).
        shading_map (numpy.ndarray): Input shading map (color image with three channels).
        upper_bounds (list): List of upper bounds for each channel [Blue, Green, Red]. If None, defaults to +22.
        penumbra_size (int): Dilation iterations for penumbra.
        show_histograms (bool): Whether to display (and return) the histograms.

    Returns:
        tuple: (median_blurred, umbra_mask, penumbra_mask, colored_mask, hist_fig)
               where hist_fig is the matplotlib Figure object (or None if show_histograms is False).
    """

    # Apply mean shift filtering
    spatial_radius = 31  # Spatial window radius
    color_radius = 25    # Color window radius
    max_level = 1        # Maximum level of the pyramid for the segmentation
    mean_shift_filtered = cv2.pyrMeanShiftFiltering(shading_map, spatial_radius, color_radius, maxLevel=max_level)

    # Apply median blur
    median_blurred = cv2.medianBlur(mean_shift_filtered, 21)

    # Separate channels
    channels = cv2.split(median_blurred)  # Split into B, G, R channels
    masks = []
    colors = ['blue', 'green', 'red']  # For histogram visualization
    channel_names = ['Blue', 'Green', 'Red']

    hist_fig = None  # Initialize the histogram figure variable

    # Create a figure with 3 vertical subplots and capture the figure object
    hist_fig, axs = plt.subplots(3, 1, figsize=(15, 8))
    hist_fig.suptitle("Channel Histograms (Blue, Green, Red)")

    for i, (channel, color, name) in enumerate(zip(channels, colors, channel_names)):
        # Calculate the histogram for the channel
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])

        ax = axs[i]
        ax.plot(hist, color=color)
        ax.set_xlim([0, 256])
        ax.set_title(f"{name} Channel Histogram")
        ax.set_xlabel("Bins")
        ax.set_ylabel("Number of Pixels")

        # Define the range for the mask
        lower_bound = int(np.min(channel))
        upper_bound = int(lower_bound + upper_bounds)

        # Create a mask for this channel
        mask = cv2.inRange(channel, lower_bound, upper_bound)
        masks.append(mask)
            
        # if show_additional_params:
        #     # Optionally display the mask in a separate window
        #     cv2.imshow(f"{name} Channel Mask", mask)

        # Add range lines to the histogram
        ax.axvline(x=lower_bound, color=color, linestyle='--', label=f'Lower Bound: {lower_bound}')
        ax.axvline(x=upper_bound, color=color, linestyle='-.', label=f'Upper Bound: {upper_bound}')
        ax.legend()

        # Adjust layout for better display
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # Instead of plt.show(), you may choose to leave it to the caller to display or save the figure.
        # plt.show()

    # Combine the masks where all pixels exist in all three channels
    umbra_mask = np.logical_and.reduce(masks).astype(np.uint8) * 255

    # Perform dilation to expand the shadow region
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    expanded_mask = cv2.dilate(umbra_mask, kernel, iterations=penumbra_size)

    # Subtract the original umbra mask from the expanded mask to create the penumbra mask
    penumbra_mask = cv2.subtract(expanded_mask, umbra_mask)

    # Create a colored output for visualization:
    umbra_colored = image.copy()
    penumbra_colored = image.copy()

    # Color the umbra in blue, but keep unshadowed areas as the original image
    umbra_colored[:, :, 0] = np.where(umbra_mask > 0, 255, image[:, :, 0])

    # Color the penumbra in red, but keep unshadowed areas as the original image
    penumbra_colored[:, :, 2] = np.where(penumbra_mask > 0, 255, image[:, :, 2])

    # Combine umbra and penumbra using bitwise OR or masking
    colored_mask = image.copy()
    colored_mask[umbra_mask > 0] = umbra_colored[umbra_mask > 0]
    colored_mask[penumbra_mask > 0] = penumbra_colored[penumbra_mask > 0]

    # Return outputs, including the histogram figure (None if show_histograms is False)
    return median_blurred, umbra_mask, penumbra_mask, colored_mask, hist_fig, masks



# Part 3: Umbra Enhancement
def umbra_enhancement(image, median_map, umbra_mask, min_scale=1.5, max_scale=2.5, epsilon=1e-5):
    """
    Enhances the umbra regions of an image using Retinex theory, applying scaling to both shadowed and unshadowed regions.

    Parameters:
        image (numpy.ndarray): Input color image (BGR format).
        median_map (numpy.ndarray): Median filtered version of the image.
        umbra_mask (numpy.ndarray): Binary mask indicating umbra regions (0 for unshadowed, >0 for shadowed).
        min_scale (float): Minimum scaling factor. Default: 1.5
        max_scale (float): Maximum scaling factor. Default: 2.5
        epsilon (float): Small constant to avoid division by zero. Default: 1e-5

    Returns:
        tuple: Enhanced image, global reference intensity (G), and local background intensity (L).
    """

    # Calculate the global reference background intensity (G) from unshadowed regions
    unshadowed_region = np.where(umbra_mask == 0)
    if len(unshadowed_region[0]) == 0:  # Check if there are unshadowed pixels
        raise ValueError("No unshadowed pixels found in the image.")
    G = np.mean(median_map[unshadowed_region], axis=0)  # Global reference intensity (RGB)

    # Calculate the local background intensity (L) from shadowed regions
    shadowed_region = np.where(umbra_mask > 0)
    if len(shadowed_region[0]) == 0:  # Check if there are shadowed pixels
        raise ValueError("No shadowed pixels found in the image.")
    L = np.mean(median_map[shadowed_region], axis=0)  # Local background intensity (RGB)

    # Enhance the umbra region by scaling based on Retinex theory
    enhanced_umbra = image.copy().astype(np.float32)

    for channel in range(3):  # Process each RGB channel separately
        scale_factor = np.clip(G[channel] / (L[channel] + epsilon), min_scale, max_scale)

        # Apply scaling and normalization in a single operation
        enhanced_umbra[:, :, channel] = np.where(
            umbra_mask > 0,  
            np.clip(image[:, :, channel] * scale_factor, None, G[channel]),  # Enhance and cap at G
            image[:, :, channel]  # Leave unshadowed regions unchanged
        )
        
    enhanced_umbra = np.clip(enhanced_umbra, 0, 255).astype(np.uint8)

    return enhanced_umbra, G, L


# Part 4: Penumbra Enhancement
def penumbra_enhancement(enhanced_umbra, penumbra_mask, G, L, min_scale=1.5, max_scale=2.5, epsilon=1e-5):
    """
    Enhance the penumbra regions of an image by applying scale factors and adaptive thresholding.

    Parameters:
    enhanced_umbra (np.ndarray): The input image with enhanced umbra regions.
    penumbra_mask (np.ndarray): A binary mask indicating the penumbra regions.
    G (np.ndarray): The global color values for each channel.
    L (np.ndarray): The local color values for each channel.
    min_scale (float): The minimum scale factor to apply. Default is 1.5.
    max_scale (float): The maximum scale factor to apply. Default is 2.5.
    epsilon (float): A small value to avoid division by zero. Default is 1e-5.

    Returns:
    np.ndarray: The final enhanced image with penumbra regions processed.
    """

    # Convert to float for precision
    enhanced_penumbra = enhanced_umbra.astype(np.float32)

    # Compute scale factors for all channels at once
    scale_factors = np.clip(G / (L + epsilon), min_scale, max_scale).reshape(1, 1, 3)

    # Apply scale factor only where penumbra mask is present
    mask = penumbra_mask > 0
    enhanced_penumbra[mask] = np.minimum(enhanced_umbra[mask] * scale_factors, G)

    # Apply adaptive thresholding on the enhanced penumbra
    binary_image = apply_adaptive_threshold(enhanced_penumbra, threshold_param=0.15)

    # Create a mask for text areas using Canny edge detection
    edges = cv2.Canny(binary_image.astype(np.uint8), 100, 200)  # Canny edge detector
    text_mask = edges > 0  # Assuming that edges correspond to text regions

    # Replace detected penumbra shadow lines with G values (vectorized operation)
    enhanced_penumbra[(binary_image == 255) & (~text_mask)] = G

    enhanced_penumbra = np.clip(enhanced_penumbra, 0, 255).astype(np.uint8)

    # Return the final enhanced image
    return enhanced_penumbra


# Main function to execute the program
if __name__ == "__main__":
    # --- Input and Ground Truth Image Initialization ---
    input_path = 'OSR_Dataset/input_images_1/INP1_B1_01.jpg' 
    input_image = cv2.imread(input_path)
    if not "NAT" in input_path:
        ground_truth_image = cv2.imread('OSR_Dataset/ground_truths_1/GT1_B1_01.jpg')

    # Start timing the execution of Enhanced LWFA Operations
    start = time.perf_counter()

    # --- Enhanced LWFA Operations ---
    start_1 = time.perf_counter() 
    shading_map = local_water_filling(input_image, max_iterations=3)      # Set max_iterations for images with bigger fonts
    end_1 = time.perf_counter()

    start_2 = time.perf_counter() 
    median_map, umbra_mask, penumbra_mask, colored_mask, histogram_figure, channel_masks = separate_umbra_and_penumbra(input_image, shading_map, upper_bounds=16, penumbra_size=2)    # Set show_histograms to help determine upper_bounds
    end_2 = time.perf_counter()

    start_3 = time.perf_counter()
    enhanced_umbra, global_color, local_color = umbra_enhancement(input_image, median_map, umbra_mask)
    end_3 = time.perf_counter()

    start_4 = time.perf_counter()
    output_image = penumbra_enhancement(enhanced_umbra, penumbra_mask, global_color, local_color)
    end_4 = time.perf_counter()

    # Stop timing after producing the output image
    end = time.perf_counter()

    # --- Evaluation Metrics ---
    # if not "NAT" in input_path:
    #     print("\n--- Evaluation Metrics Result of Enhanced LWFA Version 3 ---")
    #     print(f"Shading Map Computation Time: {end_1 - start_1:.3f}")
    #     print(f"Umbra and Penumbra Computation Time: {end_2 - start_2:.3f}")
    #     print(f"Umbra Enhancement Computation Time: {end_3 - start_3:.3f}")
    #     print(f"Penumbra Enhancement Computation Time: {end_4 - start_4:.3f}")
    #     print(f"Code Execution Time: {end - start:.3f} seconds\n")
    #     print(f'[Lower is Better] MSE: {compute_mse(ground_truth_image, output_image):.3f}')
    #     print(f'[Lower is Better] Error Ratio: {compute_error_ratio(ground_truth_image, input_image, output_image, umbra_mask):.3f}')
    #     print(f'[Closer to 1 is Better] SSIM: {compute_ssim(ground_truth_image, output_image):.3f}')
    #     print(f'[Lower is Better] DRD: {compute_drd(ground_truth_image, output_image):.3f}')
        
    #     ground_truth_text = extract_text_with_ocr(ground_truth_image)
    #     output_image_text = extract_text_with_ocr(output_image)
    #     print(f'[Lower is Better] OCR Edit Distance: {compute_ocr_accuracy(ground_truth_text, output_image_text)}\n')

    # # Display the results
    # if not "NAT" in input_path:
    #     cv2.imshow("Ground Truth Image", ground_truth_image)
    # cv2.imshow("Input Image", input_image)
    # cv2.imshow("Shading Map", shading_map)
    # cv2.imshow("Colored Mask", colored_mask)
    # cv2.imshow("Enhanced Umbra Image", enhanced_umbra)
    # cv2.imshow("Output Image", output_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
