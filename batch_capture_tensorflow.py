#!/usr/bin/env python3
"""
Batch Sediment Analysis using Original SediNet TensorFlow Models
Captures images, runs prediction via sedinet_predict1image.py, outputs D-values
"""

import numpy as np
import cv2
from PIL import Image
import time
import os
import subprocess
import atexit
import re
from datetime import datetime

# Optional imports for hardware control
try:
    import board
    import neopixel
    NEOPIXEL_AVAILABLE = True
except ImportError:
    NEOPIXEL_AVAILABLE = False

# ---------------- CONFIGURATION ----------------
MARKER_SIZE_CM = 1.4        # Your ArUco marker size (cm)
IMAGE_SIZE = 768            # SediNet expects 768x768

# Model configurations
MODELS = {
    "sand": {
        "config": "config/config_sand.json",
        "weights": "grain_size_sand_generic/res_9prcs/sand_generic_9prcs_simo_batch12_im768_768_9vars_pinball_noaug_scale.hdf5",
        "image_size": 768
    },
    "mattole": {
        "config": "config/config_mattole.json",
        "weights": "mattole/res/mattole_simo_batch7_im512_512_2vars_pinball_aug.hdf5",
        "image_size": 512
    },
    "gravel": {
        "config": "config/config_gravel.json",
        "weights": "grain_size_gravel_generic/res/gravel_generic_9prcs_simo_batch6_im768_768_9vars_pinball_aug.hdf5",
        "image_size": 768
    }
}

# NeoPixel config
LED_POWER_PIN = 24
NUM_PIXELS = 32
pixels = None

def setup_neopixel():
    """Initialize NeoPixels if available"""
    global pixels
    if not NEOPIXEL_AVAILABLE:
        return False
    try:
        pixels = neopixel.NeoPixel(board.D24, NUM_PIXELS, brightness=0.3, auto_write=False)
        atexit.register(lambda: pixels.fill((0, 0, 0)))
        print("💡 NeoPixel initialized")
        return True
    except Exception as e:
        print(f"⚠ NeoPixel setup failed: {e}")
        return False

def set_color(r, g, b):
    """Control NeoPixel colors"""
    if pixels is None:
        return
    pixels.fill((r, g, b))
    pixels.show()


class BatchSediNetTF:
    """Batch analyzer using original TensorFlow HDF5 models"""
    
    def __init__(self, model_name="sand"):
        self.model_name = model_name
        self.model_config = MODELS.get(model_name, MODELS["sand"])
        self.px_per_mm = None
        self.image_size = self.model_config["image_size"]
        
        print(f"🔧 Using SediNet model: {model_name}")
        print(f"   Config: {self.model_config['config']}")
        print(f"   Weights: {self.model_config['weights']}")
        print(f"   Image size: {self.image_size}x{self.image_size}")
        print()
    
    def draw_axis(self, frame, cam_mtx, dist, rvec, tvec, length=5.0):
        """Draw 3D coordinate axes on the marker"""
        try:
            axis = np.float32([[length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
            origin = np.float32([[0,0,0]]).reshape(-1,3)
            imgpts, _ = cv2.projectPoints(np.vstack((origin, axis)), rvec, tvec, cam_mtx, dist)
            imgpts = imgpts.astype(int)

            origin_pt = tuple(imgpts[0].ravel())
            frame = cv2.line(frame, origin_pt, tuple(imgpts[1].ravel()), (0,0,255), 2)
            frame = cv2.line(frame, origin_pt, tuple(imgpts[2].ravel()), (0,255,0), 2)
            frame = cv2.line(frame, origin_pt, tuple(imgpts[3].ravel()), (255,0,0), 2)
        except:
            pass
        return frame

    def detect_marker_scale(self, frame, marker_size_cm=MARKER_SIZE_CM):
        """Detect ArUco marker and calculate px/mm scale"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        try:
            detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
            corners, ids, _ = detector.detectMarkers(gray)
        except:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        
        px_per_mm = None
        rvec, tvec = None, None
        
        if ids is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            c = corners[0][0]
            side_px = np.mean([
                np.linalg.norm(c[0] - c[1]),
                np.linalg.norm(c[1] - c[2]),
                np.linalg.norm(c[2] - c[3]),
                np.linalg.norm(c[3] - c[0])
            ])
            
            marker_size_mm = marker_size_cm * 10
            px_per_mm = side_px / marker_size_mm
            
            cv2.putText(frame, f"Scale: {px_per_mm:.2f} px/mm", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        return px_per_mm, frame, rvec, tvec

    def run_calibration(self, cap, marker_size_cm):
        """Interactive ArUco calibration"""
        print("\n📏 CALIBRATION MODE")
        print(f"   Target Marker Size: {marker_size_cm} cm")
        print("   Controls: SPACE=accept, s=skip, r=reset points")
        
        cv2.namedWindow("Calibration")
        click_points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                click_points.append((x, y))
        
        cv2.setMouseCallback("Calibration", mouse_callback)
        
        final_scale = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detected_scale, vis_frame, _, _ = self.detect_marker_scale(frame, marker_size_cm)
            
            if detected_scale:
                final_scale = detected_scale
                status = f"DETECTED: {final_scale:.2f} px/mm"
                color = (0, 255, 0)
            else:
                status = "Looking for ArUco marker..."
                color = (0, 0, 255)
            
            cv2.putText(vis_frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw measurement points
            if len(click_points) >= 2:
                cv2.line(vis_frame, click_points[-2], click_points[-1], (255, 255, 0), 2)
                px_dist = np.linalg.norm(np.array(click_points[-2]) - np.array(click_points[-1]))
                if final_scale:
                    cm_dist = px_dist / final_scale / 10
                    cv2.putText(vis_frame, f"Distance: {cm_dist:.2f} cm", 
                               (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Calibration", vis_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32 and final_scale:  # SPACE
                print(f"✓ Calibration saved: {final_scale:.2f} px/mm")
                break
            elif key == ord('s'):
                print("⚠ Calibration skipped")
                break
            elif key == ord('r'):
                click_points = []
        
        cv2.destroyAllWindows()
        return final_scale

    def capture_images(self, num_images=10, delay_seconds=3, marker_size_cm=MARKER_SIZE_CM):
        """Capture images using webcam with calibration"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera.")
            return []
        
        set_color(255, 255, 255)  # NeoPixel ON
        
        try:
            # Calibration
            self.px_per_mm = self.run_calibration(cap, marker_size_cm)
            
            # Capture loop
            print(f"\n📸 Capturing {num_images} images...")
            image_paths = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for i in range(num_images):
                print(f"  Image {i+1}/{num_images} in {delay_seconds}s...", end='\r')
                time_end = time.time() + delay_seconds
                
                while time.time() < time_end:
                    ret, frame = cap.read()
                    if ret:
                        remaining = max(0, time_end - time.time())
                        cv2.putText(frame, f"Capturing in {remaining:.1f}s", 
                                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("Capture Preview", frame)
                        cv2.waitKey(1)
                
                # Capture
                ret, frame = cap.read()
                if ret:
                    img_name = f"sample_{timestamp}_{i+1:02d}.jpg"
                    cv2.imwrite(img_name, frame)
                    image_paths.append(img_name)
                    
                    # Save overlay
                    overlay_frame = frame.copy()
                    _, overlay_frame, _, _ = self.detect_marker_scale(overlay_frame, marker_size_cm)
                    if self.px_per_mm:
                        cv2.putText(overlay_frame, f"Scale: {self.px_per_mm:.2f} px/mm", 
                                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    overlay_dir = "stage_images"
                    os.makedirs(overlay_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(overlay_dir, f"sample_{timestamp}_{i+1:02d}_overlay.jpg"), overlay_frame)
                    
                    print(f"✓ Captured {img_name} (+ overlay)")
                else:
                    print("❌ Capture failed")
            
            cv2.destroyAllWindows()
            
        finally:
            set_color(0, 0, 0)  # NeoPixel OFF
            cap.release()
            cv2.destroyAllWindows()
        
        return image_paths

    def predict_image(self, image_path):
        """Run SediNet prediction using original TensorFlow model"""
        cmd = f'python sedinet_predict1image.py -c {self.model_config["config"]} -i {image_path} -w {self.model_config["weights"]}'
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = result.stdout + result.stderr
        
        # Parse results
        predictions = {}
        for line in output.split('\n'):
            if ':' in line:
                match = re.match(r'(P\d+|mean|sorting):\s*([\d.]+)', line.strip())
                if match:
                    key = match.group(1)
                    val = float(match.group(2))
                    predictions[key] = val
        
        return predictions

    def process_batch(self, image_paths):
        """Process all images and collect results"""
        if not image_paths:
            print("⚠ No images to process.")
            return []
        
        all_results = []
        print(f"\n⚙️  Processing {len(image_paths)} images with SediNet ({self.model_name})...\n")
        
        for i, img_path in enumerate(image_paths):
            t0 = time.time()
            
            try:
                predictions = self.predict_image(img_path)
                all_results.append(predictions)
                
                dt = time.time() - t0
                
                # Save individual result
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                result_dir = "stage_images"
                os.makedirs(result_dir, exist_ok=True)
                result_path = os.path.join(result_dir, f"{base_name}_result.txt")
                
                with open(result_path, "w") as f:
                    f.write(f"# SediNet Prediction for {img_path}\n")
                    f.write(f"# Model: {self.model_name}\n")
                    f.write(f"# Scale: {self.px_per_mm:.2f} px/mm\n" if self.px_per_mm else "")
                    f.write("# Units: micrometers (µm)\n\n")
                    for k, v in sorted(predictions.items()):
                        f.write(f"{k}: {v:.4f} µm  ({v/1000:.4f} mm)\n")
                
                # Progress
                bar_len = 30
                filled = int(bar_len * (i+1) // len(image_paths))
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"  [{bar}] {i+1}/{len(image_paths)} ✓ {os.path.basename(img_path)} ({dt:.2f}s)")
                
            except Exception as e:
                print(f"  ❌ Error processing {os.path.basename(img_path)}: {e}")
        
        print(f"\n✓ Processing complete!")
        return all_results

    def calculate_d_values(self, avg_percentiles):
        """Convert P-values to D-values"""
        p_keys = ['P5', 'P10', 'P16', 'P25', 'P50', 'P75', 'P84', 'P90', 'P95']
        p_positions = [5, 10, 16, 25, 50, 75, 84, 90, 95]
        # Restored full D-value range including D90
        d_positions = [10, 30, 50, 70, 90]
        
        p_values = [avg_percentiles.get(k, 0) for k in p_keys]
        d_values = np.interp(d_positions, p_positions, p_values)
        
        return dict(zip([f'D{p}' for p in d_positions], d_values))

    def analyze_batch(self, image_paths):
        """Complete batch analysis"""
        all_results = self.process_batch(image_paths)
        
        if not all_results:
            return None, None, None, []
        
        # Average results
        avg_percentiles = {}
        std_percentiles = {}
        
        all_keys = set()
        for r in all_results:
            all_keys.update(r.keys())
        
        for key in all_keys:
            values = [r.get(key, 0) for r in all_results if key in r]
            if values:
                avg_percentiles[key] = np.mean(values)
                std_percentiles[key] = np.std(values)
        
        d_values = self.calculate_d_values(avg_percentiles)
        
        return avg_percentiles, std_percentiles, d_values, all_results


def print_results(avg_percentiles, std_percentiles, d_values):
    """Display final results"""
    if avg_percentiles is None:
        print("❌ No results to display")
        return
    
    print("\n" + "="*60)
    print("           SEDIMENT ANALYSIS RESULTS")
    print("="*60)
    
    print("\nAveraged Percentiles (µm):")
    print("-"*60)
    for key in ['P5', 'P10', 'P16', 'P25', 'P50', 'P75', 'P84', 'P90', 'P95']:
        if key in avg_percentiles:
            avg = avg_percentiles[key]
            std = std_percentiles.get(key, 0)
            print(f"  {key:4s}: {avg/1000:8.4f} mm  (±{std/1000:.4f})")
    
    print("\n" + "="*60)
    print("           D-VALUES (Key Grain Sizes)")
    print("="*60)
    for key in ['D10', 'D30', 'D50', 'D70', 'D90']:
        if key in d_values:
            print(f"  {key}: {d_values[key]/1000:.4f} mm")
    
    print("="*60)


def main():
    print("\n" + "="*60)
    print("   SEDIMENT BATCH ANALYSIS - TensorFlow/HDF5 System")
    print("="*60 + "\n")
    
    # Model selection
    print("Available models:")
    print("  1. sand     - Sand (9 percentiles, 768x768)")
    print("  2. mattole  - Mixed sand/gravel (mean+sorting, 512x512)")
    print("  3. gravel   - Gravel (9 percentiles, 768x768)")
    
    choice = input("\nSelect model (1/2/3) [default: 1]: ").strip() or "1"
    model_map = {"1": "sand", "2": "mattole", "3": "gravel"}
    model_name = model_map.get(choice, "sand")
    
    # Marker size
    marker_input = input(f"\nEnter ArUco marker size in cm [default: {MARKER_SIZE_CM}]: ").strip()
    marker_size = float(marker_input) if marker_input else MARKER_SIZE_CM
    
    # Number of images
    num_input = input("\nNumber of images to capture [default: 10]: ").strip()
    num_images = int(num_input) if num_input else 10
    
    # Initialize
    setup_neopixel()
    analyzer = BatchSediNetTF(model_name)
    
    # Capture images
    image_paths = analyzer.capture_images(num_images, delay_seconds=3, marker_size_cm=marker_size)
    
    # Analyze
    avg_percentiles, std_percentiles, d_values, all_results = analyzer.analyze_batch(image_paths)
    
    # Print results
    print_results(avg_percentiles, std_percentiles, d_values)
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"sediment_analysis_{timestamp}.txt"
    
    with open(summary_path, "w") as f:
        f.write("SEDIMENT ANALYSIS SUMMARY\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Images: {len(image_paths)}\n\n")
        
        if d_values:
            f.write("D-VALUES (mm):\n")
            for k, v in d_values.items():
                f.write(f"  {k}: {v/1000:.4f}\n")
    
    print(f"\n💾 Results saved to: {summary_path}")
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
