#ocr_utils.py
import re
import cv2
from config import MODEL_PATH
from ultralytics import YOLO
from paddleocr import PaddleOCR

model = YOLO(MODEL_PATH)
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
def smart_correct_ocr_text(text, registered_plates=None):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(text) < 9 or len(text) > 10:
        return text

    if text.startswith("HO"):
        text = "JH" + text[3:]
    elif text.startswith("H0"):
        text = "JH" + text[3:]
    elif text.startswith("H"):
        text = "JH" + text[2:]
    elif text[0] == "J" and text[1] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        possible_states = ["OD", "OR", "ON", "OP", "OA", "OB", "OC", "OJ"] 
        candidate = "O" + text[1:]
        if candidate[:2] in possible_states:
            text = candidate

    corrected = []
    for i, c in enumerate(text):
        if i < 2:
            corrected.append(c)
        elif i < 4:
            if c in ['O', 'Q']:
                corrected.append('0')
            elif c in ['I', 'L', '|']:
                corrected.append('1')
            elif c in ['S']:
                corrected.append('5')
            elif c in ['G']:
                corrected.append('6')
            else:
                corrected.append(c)
        elif i < 6:
            corrected.append(c)
        else:
            if c in ['O', 'Q']:
                corrected.append('0')
            elif c in ['I', 'L', '|']:
                corrected.append('1')
            elif c in ['S']:
                corrected.append('5')
            elif c in ['G']:
                corrected.append('6')
            else:
                corrected.append(c)
    candidate = ''.join(corrected)

    if registered_plates:
        if candidate in registered_plates:
            return candidate
        series = candidate[4:6]
        swapped = []
        for a, b in [(series.replace('O', 'D'), series.replace('D', 'O'))]:
            test = candidate[:4] + a + candidate[6:]
            if test in registered_plates:
                return test
            test = candidate[:4] + b + candidate[6:]
            if test in registered_plates:
                return test
        return candidate
    else:
        return candidate

def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    eq = cv2.equalizeHist(blur)
    thresh = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

def try_ocr_with_retries(image, ocr_model):
    try:
        variants = [
            preprocess_for_ocr(image),
            cv2.rotate(preprocess_for_ocr(image), cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(preprocess_for_ocr(image), cv2.ROTATE_90_COUNTERCLOCKWISE)
        ]
        for variant in variants:
            result = ocr_model.ocr(variant, cls=True)
            if result and isinstance(result, list) and len(result) > 0 and len(result[0]) > 0:
                return result
    except Exception:
        return None
    return None