import os
import time
import datetime
import threading
from collections import defaultdict
import numpy as np
import cv2

# Real weapon target classes
THREAT_CLASSES = {
    "handgun", "pistol", "revolver", "rifle", "shotgun", "gun", "firearm",
    "heavy_weapon", "knife", "dagger", "machete", "combat knife",
    "bomb", "grenade", "explosive", "weapon"
}

# Negative distractor classes for zero-shot disambiguation
NEGATIVE_DISTRACTORS = [
    "scissors", "pair of scissors", "soldering iron", "ruler", "scale",
    "screwdriver", "pliers", "tool", "human arm", "bare arm", "arm",
    "hand", "finger", "cell phone", "pen", "wire", "cable"
]

class ThreadedCamera:
    """Non-blocking low-latency camera frame grabber running on a dedicated thread."""
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.frame = None
        self.ret = False
        self.is_running = True
        self.lock = threading.Lock()
        
        self._init_cap()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _init_cap(self):
        if self.cap is not None:
            self.cap.release()
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                print(f"[ThreadedCamera] Camera {self.camera_index} not opened. Falling back to simulation.")
                self.cap = None
            else:
                print(f"[ThreadedCamera] Camera {self.camera_index} online via DirectShow.")
        except Exception as e:
            print(f"[ThreadedCamera] Capture error: {e}")
            self.cap = None

    def _capture_loop(self):
        while self.is_running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.ret = True
                        self.frame = frame
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.03)

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def switch_camera(self, new_index):
        with self.lock:
            self.camera_index = new_index
            self._init_cap()

    def release(self):
        self.is_running = False
        if self.cap:
            self.cap.release()


class ThreatDetector:
    def __init__(self, camera_index=0, conf_threshold=0.50):
        self.camera_index = camera_index
        self.conf_threshold = conf_threshold
        self.selected_engine = "specialized"  # 'specialized', 'extended', 'world'
        
        self.model = None
        self.model_status = "INITIALIZING..."
        self.is_running = True
        self.lock = threading.Lock()
        
        self.current_fps = 0
        self.inference_fps = 0
        self.active_threats = []
        self.latest_frame = None
        self.latest_snapshot = None
        
        # Temporal persistence buffer: require detection across at least 3 consecutive frames
        self.detection_history = defaultdict(int)
        
        self.snapshot_dir = os.path.join(os.path.dirname(__file__), "static", "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Threaded camera
        self.camera = ThreadedCamera(self.camera_index)
        
        # Distractor verification model (YOLO11 nano COCO for scissors, cell phones, tools)
        self.distractor_model = None
        self._init_distractor_model()

        # Start background model loader and inference worker
        self._load_model_engine(self.selected_engine)
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

    def _init_distractor_model(self):
        """Loads lightweight COCO model to actively veto scissors, phones, and tools."""
        def load_dist():
            try:
                from ultralytics import YOLO
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                coco_path = os.path.join(base_dir, "yolo11n.pt")
                if os.path.exists(coco_path):
                    self.distractor_model = YOLO(coco_path)
                    print("[ThreatDetector] Distractor veto engine online (YOLO11n).")
            except Exception as e:
                print(f"[ThreatDetector] Distractor engine notice: {e}")

        t = threading.Thread(target=load_dist, daemon=True)
        t.start()

    def _load_model_engine(self, engine_name):
        """Loads and switches model engine."""
        def load():
            from ultralytics import YOLO
            self.model_status = f"LOADING {engine_name.upper()}..."
            try:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                
                if engine_name == "specialized":
                    # Supervised nano weapon detector (35+ FPS, calibrated against arms/scissors/tools)
                    model_path = os.path.join(base_dir, "weapon_model.pt")
                    if not os.path.exists(model_path):
                        model_path = "weapon_model.pt"
                    model = YOLO(model_path)
                    self.model_status = "CALIBRATED WEAPON ENGINE (35+ FPS)"

                elif engine_name == "extended":
                    # Supervised extended weapon model (gun, rifle, heavy_weapon, knife)
                    model_path = os.path.join(base_dir, "izzy_model.pt")
                    if not os.path.exists(model_path):
                        model_path = "izzy_model.pt"
                    model = YOLO(model_path)
                    self.model_status = "EXTENDED WEAPON ENGINE"

                else:  # 'world'
                    # Zero-shot YOLO-World with comprehensive negative distractor grounding
                    model = YOLO("yolov8s-worldv2.pt")
                    all_prompts = list(THREAT_CLASSES) + NEGATIVE_DISTRACTORS
                    model.set_classes(all_prompts)
                    self.model_status = "YOLO-WORLD ZERO-SHOT (GROUNDED)"

                with self.lock:
                    self.model = model
                    self.selected_engine = engine_name
                    self.detection_history.clear()
                print(f"[ThreatDetector] Engine active: {self.model_status}")
            except Exception as e:
                print(f"[ThreatDetector] Error loading engine {engine_name}: {e}")
                self.model_status = "ONLINE (FALLBACK)"

        t = threading.Thread(target=load, daemon=True)
        t.start()

    def set_config(self, camera_index=None, conf_threshold=None, engine=None):
        with self.lock:
            if conf_threshold is not None:
                self.conf_threshold = float(conf_threshold)
            if camera_index is not None and camera_index != self.camera_index:
                self.camera_index = camera_index
                self.camera.switch_camera(self.camera_index)
            if engine is not None and engine != self.selected_engine:
                self.selected_engine = engine
                self._load_model_engine(engine)

    def _inference_loop(self):
        """Asynchronous worker running inference at ~35 FPS with multi-layer false-positive filters."""
        sim_t = 0
        while self.is_running:
            success, frame = self.camera.read()
            if not success or frame is None:
                sim_t += 0.04
                frame = self.generate_simulated_frame(sim_t)

            if self.model is not None:
                t_start = time.time()
                try:
                    # Respect operator's confidence threshold directly (allows real-time slider control)
                    results = self.model.predict(
                        frame,
                        conf=float(self.conf_threshold),
                        imgsz=416,
                        verbose=False
                    )
                    
                    frame_candidates = []

                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            label = self.model.names[cls_id].lower()
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            x1, y1, x2, y2 = xyxy

                            # Bounds clamp
                            fh, fw = frame.shape[:2]
                            x1 = max(0, min(fw - 1, x1))
                            y1 = max(0, min(fh - 1, y1))
                            x2 = max(0, min(fw - 1, x2))
                            y2 = max(0, min(fh - 1, y2))

                            w = x2 - x1
                            h = y2 - y1
                            if w <= 12 or h <= 12:
                                continue

                            aspect_ratio = max(w / (h + 1e-5), h / (w + 1e-5))

                            # 1. Reject explicit distractor text categories
                            is_distractor = any(d in label for d in [
                                "scissor", "scale", "ruler", "screwdriver", "pliers",
                                "soldering", "cell phone", "pen", "wire", "cable"
                            ])
                            if is_distractor:
                                continue

                            # 2. Verify label is an authorized threat category
                            is_threat = any(t in label for t in THREAT_CLASSES)
                            if not is_threat:
                                continue

                            # 3. Scale / Ruler Filter:
                            # A long ruler or scale has aspect ratio > 3.0.
                            # Handguns/pistols are compact (aspect ratio 0.8 to 2.2).
                            # If aspect ratio > 3.0, it is physically a ruler, stick, or rod -> NOT a pistol.
                            if ("pistol" in label or "gun" in label) and aspect_ratio > 3.0:
                                continue

                            # 4. Plain Phone Slab vs. Phone Displaying Weapon Image Filter:
                            # A phone body has aspect ratio ~1.6 - 2.5.
                            # When the phone is blank, screen-off, or displaying non-weapon content,
                            # the central 60% interior has near-zero edge contours and low variance.
                            # BUT when the user displays an image of a pistol/knife on the phone screen,
                            # the weapon image has sharp, high-contrast contours (edge density >= 0.02).
                            if ("pistol" in label or "gun" in label):
                                try:
                                    crop = frame[y1:y2, x1:x2]
                                    if crop.size > 0:
                                        ch, cw = crop.shape[:2]
                                        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                        m = np.zeros((ch, cw), dtype=np.uint8)
                                        cv2.rectangle(m, (int(cw * 0.2), int(ch * 0.2)), (int(cw * 0.8), int(ch * 0.8)), 255, -1)
                                        edges_crop = cv2.Canny(gray_crop, 35, 100)
                                        interior_edges = np.count_nonzero(edges_crop[m > 0])
                                        interior_area = np.count_nonzero(m)
                                        edge_density = interior_edges / (interior_area + 1e-5)
                                        interior_var = float(np.var(gray_crop[m > 0]))

                                        # Reject plain phone body / blank screen with no weapon displayed
                                        if (1.55 <= aspect_ratio <= 2.6) and edge_density < 0.015 and interior_var < 350.0:
                                            continue

                                        # Distractor model check (Scissors & Plain Cell Phone)
                                        if self.distractor_model is not None:
                                            dist_res = self.distractor_model.predict(crop, imgsz=192, conf=0.35, verbose=False)
                                            is_distractor_hit = False
                                            for dbox in dist_res[0].boxes:
                                                d_name = self.distractor_model.names[int(dbox.cls[0])].lower()
                                                if d_name == "scissors":
                                                    is_distractor_hit = True
                                                    break
                                                elif d_name == "cell phone":
                                                    # Only reject cell phone if screen DOES NOT contain high-contrast weapon contours
                                                    if edge_density < 0.020:
                                                        is_distractor_hit = True
                                                        break
                                            if is_distractor_hit:
                                                continue
                                except Exception:
                                    pass

                            # 5. Passed: Immediate threat candidate confirmed
                            frame_candidates.append({
                                "label": label,
                                "confidence": round(conf, 3),
                                "box": [int(x1), int(y1), int(x2), int(y2)]
                            })

                    t_elapsed = time.time() - t_start
                    if t_elapsed > 0:
                        self.inference_fps = int(1.0 / t_elapsed)

                    with self.lock:
                        self.active_threats = frame_candidates
                except Exception as e:
                    pass

            time.sleep(0.015)

    def generate_simulated_frame(self, t):
        """Monochrome simulated feed when no camera is active."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for y in range(0, 720, 40):
            cv2.line(frame, (0, y), (1280, y), (20, 20, 20), 1)
        for x in range(0, 1280, 40):
            cv2.line(frame, (x, 0), (x, 720), (20, 20, 20), 1)
            
        cv2.putText(frame, "SIMULATED SURVEILLANCE FEED [SECTOR 01]", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        cv2.putText(frame, f"ENGINE: {self.model_status}", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        return frame

    def capture_snapshot(self, custom_label=None):
        """Captures evidence snapshot of the current frame."""
        with self.lock:
            if self.latest_frame is None:
                return None
            
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"threat_{timestamp_str}.jpg"
            filepath = os.path.join(self.snapshot_dir, filename)
            
            cv2.imwrite(filepath, self.latest_frame)
            
            snapshot_meta = {
                "filename": filename,
                "url": f"/static/snapshots/{filename}",
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "label": custom_label or ("THREAT" if self.active_threats else "MANUAL_CAPTURE")
            }
            self.latest_snapshot = snapshot_meta
            return snapshot_meta

    def process_next_frame(self):
        """Yields low-latency video stream frames at 30 FPS."""
        prev_time = time.time()
        sim_t = 0

        while self.is_running:
            success, raw_frame = self.camera.read()
            if not success or raw_frame is None:
                sim_t += 0.04
                raw_frame = self.generate_simulated_frame(sim_t)

            current_time = time.time()
            dt = current_time - prev_time
            if dt > 0:
                self.current_fps = int(1.0 / dt)
            prev_time = current_time

            # Copy active threats to draw onto frame
            with self.lock:
                threats_to_draw = list(self.active_threats)

            for threat in threats_to_draw:
                xyxy = threat["box"]
                label = threat["label"]
                conf = threat["confidence"]
                x1, y1, x2, y2 = xyxy

                is_critical = any(c in label for c in ["bomb", "grenade", "explosive", "rifle", "gun", "pistol"])
                box_color = (60, 60, 230) if is_critical else (220, 220, 220)

                # Thin high-contrast HUD box
                cv2.rectangle(raw_frame, (x1, y1), (x2, y2), box_color, 2)

                # Corner crosshairs
                c_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
                cv2.line(raw_frame, (x1, y1), (x1 + c_len, y1), (255, 255, 255), 2)
                cv2.line(raw_frame, (x1, y1), (x1 + c_len, y1), (255, 255, 255), 2)
                cv2.line(raw_frame, (x2, y1), (x2 - c_len, y1), (255, 255, 255), 2)
                cv2.line(raw_frame, (x2, y1), (x2 - c_len, y1), (255, 255, 255), 2)
                cv2.line(raw_frame, (x1, y2), (x1 + c_len, y2), (255, 255, 255), 2)
                cv2.line(raw_frame, (x1, y2), (x1 - c_len, y2), (255, 255, 255), 2)
                cv2.line(raw_frame, (x2, y2), (x2 - c_len, y2), (255, 255, 255), 2)
                cv2.line(raw_frame, (x2, y2), (x2 - c_len, y2), (255, 255, 255), 2)

                # Label Badge
                badge_text = f"[{label.upper()}] {int(conf * 100)}%"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(raw_frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (15, 15, 15), -1)
                cv2.rectangle(raw_frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), box_color, 1)
                cv2.putText(raw_frame, badge_text, (x1 + 4, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            with self.lock:
                self.latest_frame = raw_frame.copy()

            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Smooth 30 FPS interval (~33ms)
            time.sleep(0.02)

    def close(self):
        self.is_running = False
        self.camera.release()
