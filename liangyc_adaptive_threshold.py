'''
The Python implementation of Bradley and Roth's Adaptive Thresholding using Integral Image.
Credit to Liang-yc: https://github.com/Liang-yc/IntegralThreshold
'''

import cv2
import numpy as np
from numba import jit

@jit(nopython=True)
def thresholdIntegral(inputMat, s, T = 0.15):
    
    # Initialize with white background (255)
    outputMat = np.full(inputMat.shape, 255)
    nRows = inputMat.shape[0]
    nCols = inputMat.shape[1]
    S = int(max(nRows, nCols) / 8)
    s2 = int(S / 4)
    
    for i in range(nRows):
        y1 = max(0, i - s2)
        y2 = min(nRows - 1, i + s2)
        
        for j in range(nCols):
            x1 = max(0, j - s2)
            x2 = min(nCols - 1, j + s2)
            
            count = (x2 - x1)*(y2 - y1)
            sum = s[y2][x2] - s[y2][x1] - s[y1][x2] + s[y1][x1]
            
            if ((int)(inputMat[i][j] * count) < (int)(sum*(1.0 - T))):
                outputMat[i][j] = 0  # Set text to black (0)
                
    return outputMat

def apply_adaptive_threshold(image, threshold_param=0.15, invert=True):
    
    # Ensure image is grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute integral image
    integral = cv2.integral(image)
    
    # Apply thresholding
    thresh = thresholdIntegral(image, integral, threshold_param)
    
    # Convert to uint8
    thresh = np.uint8(thresh)
    
    # Invert if needed
    if not invert:  # Keep original black background, white text
        thresh = cv2.bitwise_not(thresh)
    
    return thresh

# Example usage
if __name__ == '__main__':
    # Your test code here
    image = cv2.imread('testimage.jpg', 0)
    
    # Get white background, black text
    thresh_inverted = apply_adaptive_threshold(image, threshold_param=0.15, invert=True)
    
    # Get black background, white text
    thresh_original = apply_adaptive_threshold(image, threshold_param=0.15, invert=False)
    
    # Display results
    cv2.imshow('White Background, Black Text', thresh_inverted)
    cv2.imshow('Black Background, White Text', thresh_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()