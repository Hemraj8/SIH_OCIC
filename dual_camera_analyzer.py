#!/usr/bin/env python3
"""
Dual-Camera Grain Size Analysis System
Combines:
1. Webcam (1080p @ 15cm) → SediNet ML prediction
2. Microscopic Camera (640x480) → CV Segmentation
3. Result Fusion → Averaged D-values + Logarithmic plot
"""

import cv2
import numpy as np
import os
import subprocess
import json
from datetime import datetime
import matplotlib.pyplot as plt

# ArUco marker for calibration
MARKER_SIZE_CM = 1.4  # Your printed ArUco marker size

class DualCameraAnalyzer:
    def __init__(self, conda_env="sedinet_m1"):
        self.conda_env = conda_env
        self.webcam_px_per_mm = None
        self.microscope_px_per_mm = None
        
        # SediNet model paths
        self.sand_config = "config/config_sand.json"
        self.sand_weights = "grain_size_sand_generic/res_9prcs/sand_generic_9prcs_simo_batch12_im768_768_9vars_pinball_noaug_scale.hdf5"
        self.mattole_config = "config/config_mattole.json"
        self.mattole_weights = "mattole/res/mattole_simo_batch7_im512_512_2vars_pinball_aug.hdf5"
        
    # ============ WEBCAM (ML) SECTION ============
    def calibrate_webcam(self, camera_index=0):
        """ArUco-based calibration for webcam"""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        print("📷 Webcam Calibration - Show ArUco marker, press SPACE to accept")
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            try:
                detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
                corners, ids, _ = detector.detectMarkers(gray)
            except:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
            
            if ids is not None and len(corners) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                # Calculate px/mm
                c = corners[0][0]
                side_px = np.linalg.norm(c[0] - c[1])
                self.webcam_px_per_mm = side_px / (MARKER_SIZE_CM * 10)
                cv2.putText(frame, f"Scale: {self.webcam_px_per_mm:.2f} px/mm", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Webcam Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32 and self.webcam_px_per_mm:  # SPACE
                break
            elif key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"✓ Webcam calibrated: {self.webcam_px_per_mm:.2f} px/mm")
        return self.webcam_px_per_mm
    
    def capture_webcam_image(self, camera_index=0, save_path="webcam_capture.jpg"):
        """Capture single image from webcam"""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(save_path, frame)
            print(f"✓ Webcam image saved: {save_path}")
        cap.release()
        return save_path if ret else None
    
    def predict_with_sedinet(self, image_path, model="sand"):
        """Run SediNet prediction using original TensorFlow models"""
        if model == "sand":
            config = self.sand_config
            weights = self.sand_weights
        else:
            config = self.mattole_config
            weights = self.mattole_weights
        
        cmd = f'conda run -n {self.conda_env} python sedinet_predict1image.py -c {config} -i {image_path} -w {weights}'
        
        print(f"🧠 Running SediNet ({model})...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Parse output
        output = result.stdout + result.stderr
        predictions = {}
        for line in output.split('\n'):
            if ':' in line and any(p in line for p in ['P5', 'P10', 'P16', 'P25', 'P50', 'P75', 'P84', 'P90', 'P95', 'mean', 'sorting']):
                parts = line.strip().split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    try:
                        val = float(parts[1].strip())
                        predictions[key] = val
                    except:
                        pass
        
        return predictions
    
    # ============ MICROSCOPE (CV) SECTION ============
    def calibrate_microscope(self, camera_index=1):
        """Calibrate microscopic camera using known object"""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("🔬 Microscope Calibration - Place a known-size object")
        print("   Draw a line across it and enter the real length")
        
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
        
        cv2.namedWindow("Microscope Calibration")
        cv2.setMouseCallback("Microscope Calibration", mouse_callback)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw points/line
            for p in points:
                cv2.circle(frame, p, 5, (0, 255, 0), -1)
            if len(points) == 2:
                cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
                px_dist = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
                cv2.putText(frame, f"Pixel Distance: {px_dist:.1f} px", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, "Click 2 points, press ENTER when done", 
                       (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Microscope Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13 and len(points) == 2:  # ENTER
                break
            elif key == ord('r'):
                points = []
            elif key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(points) == 2:
            px_dist = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
            real_mm = float(input("Enter the real length in mm: "))
            self.microscope_px_per_mm = px_dist / real_mm
            print(f"✓ Microscope calibrated: {self.microscope_px_per_mm:.2f} px/mm")
        
        return self.microscope_px_per_mm
    
    def capture_microscope_image(self, camera_index=1, save_path="microscope_capture.jpg"):
        """Capture single image from microscope"""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(save_path, frame)
            print(f"✓ Microscope image saved: {save_path}")
        cap.release()
        return save_path if ret else None
    
    def segment_grains_cv(self, image_path):
        """Segment sand grains using OpenCV watershed"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Sure foreground using distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Watershed
        markers = cv2.watershed(img, markers)
        
        # Find contours for each grain
        grain_sizes = []
        for label in range(2, markers.max() + 1):
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[markers == label] = 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                area = cv2.contourArea(contours[0])
                if area > 10:  # Filter noise
                    equiv_diameter_px = np.sqrt(4 * area / np.pi)
                    grain_sizes.append(equiv_diameter_px)
        
        # Convert to mm if calibrated
        if self.microscope_px_per_mm:
            grain_sizes_mm = [s / self.microscope_px_per_mm for s in grain_sizes]
        else:
            grain_sizes_mm = grain_sizes  # Still in pixels
        
        return grain_sizes_mm, markers, img
    
    def calculate_percentiles_cv(self, grain_sizes_mm):
        """Calculate D-values from grain size distribution"""
        if not grain_sizes_mm:
            return {}
        
        sizes = np.array(grain_sizes_mm) * 1000  # Convert to micrometers
        percentiles = {
            'P5': np.percentile(sizes, 5),
            'P10': np.percentile(sizes, 10),
            'P16': np.percentile(sizes, 16),
            'P25': np.percentile(sizes, 25),
            'P50': np.percentile(sizes, 50),
            'P75': np.percentile(sizes, 75),
            'P84': np.percentile(sizes, 84),
            'P90': np.percentile(sizes, 90),
            'P95': np.percentile(sizes, 95),
        }
        return percentiles
    
    # ============ FUSION & PLOTTING ============
    def fuse_results(self, ml_results, cv_results, weight_ml=0.6, weight_cv=0.4):
        """Average results from ML and CV methods"""
        fused = {}
        all_keys = set(list(ml_results.keys()) + list(cv_results.keys()))
        
        for key in all_keys:
            ml_val = ml_results.get(key)
            cv_val = cv_results.get(key)
            
            if ml_val is not None and cv_val is not None:
                fused[key] = weight_ml * ml_val + weight_cv * cv_val
            elif ml_val is not None:
                fused[key] = ml_val
            elif cv_val is not None:
                fused[key] = cv_val
        
        return fused
    
    def plot_distribution(self, ml_results, cv_results, fused_results, save_path="grain_distribution.png"):
        """Create logarithmic cumulative distribution plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        percentile_order = ['P5', 'P10', 'P16', 'P25', 'P50', 'P75', 'P84', 'P90', 'P95']
        percentile_values = [5, 10, 16, 25, 50, 75, 84, 90, 95]
        
        def extract_sizes(results):
            sizes = []
            for p in percentile_order:
                if p in results:
                    sizes.append(results[p])
                else:
                    sizes.append(np.nan)
            return sizes
        
        ml_sizes = extract_sizes(ml_results)
        cv_sizes = extract_sizes(cv_results)
        fused_sizes = extract_sizes(fused_results)
        
        # Plot
        if any(not np.isnan(s) for s in ml_sizes):
            ax.semilogx(ml_sizes, percentile_values, 'b-o', label='ML (SediNet)', linewidth=2)
        if any(not np.isnan(s) for s in cv_sizes):
            ax.semilogx(cv_sizes, percentile_values, 'g-s', label='CV (Segmentation)', linewidth=2)
        if any(not np.isnan(s) for s in fused_sizes):
            ax.semilogx(fused_sizes, percentile_values, 'r-^', label='Fused (Average)', linewidth=2, markersize=8)
        
        ax.set_xlabel('Grain Size (µm)', fontsize=12)
        ax.set_ylabel('Cumulative % Finer', fontsize=12)
        ax.set_title('Grain Size Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"📊 Distribution plot saved: {save_path}")
        plt.show()
    
    def run_full_analysis(self):
        """Complete dual-camera analysis workflow"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*60)
        print("   DUAL-CAMERA GRAIN SIZE ANALYSIS")
        print("="*60 + "\n")
        
        # Step 1: Calibrate cameras
        print("Step 1/5: Calibrating webcam...")
        self.calibrate_webcam(camera_index=0)
        
        print("\nStep 2/5: Calibrating microscope...")
        self.calibrate_microscope(camera_index=1)
        
        # Step 3: Capture images
        print("\nStep 3/5: Capturing images...")
        webcam_img = self.capture_webcam_image(save_path=f"webcam_{timestamp}.jpg")
        micro_img = self.capture_microscope_image(save_path=f"microscope_{timestamp}.jpg")
        
        # Step 4: Run analyses
        print("\nStep 4/5: Running analyses...")
        
        # ML prediction
        ml_results = self.predict_with_sedinet(webcam_img, model="sand")
        print(f"  ML Results: {ml_results}")
        
        # CV segmentation
        grain_sizes, _, _ = self.segment_grains_cv(micro_img)
        cv_results = self.calculate_percentiles_cv(grain_sizes)
        print(f"  CV Results: {cv_results}")
        
        # Step 5: Fuse and plot
        print("\nStep 5/5: Fusing results and plotting...")
        fused = self.fuse_results(ml_results, cv_results)
        print(f"  Fused Results: {fused}")
        
        self.plot_distribution(ml_results, cv_results, fused, save_path=f"distribution_{timestamp}.png")
        
        # Print D-values
        print("\n" + "="*60)
        print("   FINAL D-VALUES (Fused)")
        print("="*60)
        for key in ['P10', 'P30', 'P50', 'P70', 'P90']:
            if key in fused:
                print(f"  D{key[1:]}: {fused[key]/1000:.4f} mm")
        
        return fused


if __name__ == "__main__":
    analyzer = DualCameraAnalyzer()
    
    print("Choose mode:")
    print("1. Full dual-camera analysis")
    print("2. Test SediNet only (webcam)")
    print("3. Test CV segmentation only (microscope)")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        analyzer.run_full_analysis()
    elif choice == "2":
        analyzer.calibrate_webcam()
        img = analyzer.capture_webcam_image()
        results = analyzer.predict_with_sedinet(img, model="sand")
        print(f"\nSediNet Results: {results}")
    elif choice == "3":
        analyzer.calibrate_microscope()
        img = analyzer.capture_microscope_image()
        sizes, _, _ = analyzer.segment_grains_cv(img)
        results = analyzer.calculate_percentiles_cv(sizes)
        print(f"\nCV Results: {results}")
