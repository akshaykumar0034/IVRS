# detection_utils.py
import re
from config import CONFIDENCE_THRESHOLD

def expand_box(box, image_shape, margin=0.05, min_margin=5):
    x1, y1, x2, y2 = box
    h, w = image_shape[:2]
    dx = max(int((x2 - x1) * margin), min_margin)
    dy = max(int((y2 - y1) * margin), min_margin)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)
    return x1, y1, x2, y2

def process_frame(frame, model, ocr_model, plate_pattern, CONFIDENCE_THRESHOLD, smart_correct_ocr_text, try_ocr_with_retries, target_plate=None, last_box=None):
    results = model(frame)
    found = False
    current_box = last_box
    frame_plates = []
    boxes = results[0].boxes
    if hasattr(boxes, 'conf'):
        for i, box in enumerate(boxes.xyxy):
            conf = float(boxes.conf[i])
            if conf < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box)
            x1, y1, x2, y2 = expand_box((x1, y1, x2, y2), frame.shape, margin=0.05)
            cropped = frame[y1:y2, x1:x2]
            if cropped.shape[0] < 20 or cropped.shape[1] < 60:
                continue
            ocr_result = try_ocr_with_retries(cropped, ocr_model)
            if not ocr_result:
                continue
            text = ''.join([line[1][0] for line in ocr_result[0]]).upper()
            cleaned = re.sub(r'[^A-Z0-9]', '', text)
            corrected = smart_correct_ocr_text(cleaned)
            if re.match(plate_pattern, corrected):
                frame_plates.append(corrected)
                if corrected == target_plate:
                    current_box = (x1, y1, x2, y2)
                    found = True
                    break
    return frame, frame_plates, current_box, found
