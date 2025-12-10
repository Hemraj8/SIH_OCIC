#!/usr/bin/env python3
"""
Generate a specific ArUco marker (ID 0, DICT_4X4_50) for SediNet calibration.
Run this script to save 'aruco_marker_0.png'.
"""
import cv2
import sys

def main():
    # Configuration
    MARKER_ID = 0
    MARKER_SIZE_PIXELS = 600  # High res for printing
    OUTPUT_FILE = f"aruco_marker_{MARKER_ID}.png"
    
    print(f"Generating ArUco Marker ID {MARKER_ID}...")
    
    try:
        # Try new OpenCV API (4.6+)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_img = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, MARKER_SIZE_PIXELS)
    except AttributeError:
        # Fallback for older OpenCV
        try:
            import cv2.aruco as aruco
            dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            marker_img = aruco.drawMarker(dictionary, MARKER_ID, MARKER_SIZE_PIXELS)
        except Exception as e:
            print(f"Error accessing ArUco: {e}")
            print("Please ensure opencv-contrib-python is installed.")
            sys.exit(1)
            
    # Save
    cv2.imwrite(OUTPUT_FILE, marker_img)
    print(f"✓ Saved to: {OUTPUT_FILE}")
    print("  -> Print this image ensuring it is square.")
    print("  -> Measure the side length in cm (e.g., 2.5 cm) for calibration.")

if __name__ == "__main__":
    main()
