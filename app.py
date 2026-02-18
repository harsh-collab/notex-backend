from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import time, io, uuid
import os, json
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import easyocr
from spellchecker import SpellChecker
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# Initialize EasyOCR reader for handwriting recognition
print("Loading EasyOCR model...")
ocr_reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if CUDA available
print("EasyOCR model loaded successfully!")

# Initialize SpellChecker
print("Loading SpellChecker...")
spell = SpellChecker()
print("SpellChecker loaded successfully!")

app = Flask(__name__)
CORS(app)  # allow requests from frontend

UPLOAD_DIR = r"D:\capstone_project - Copy\notex-backend\uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")

# Custom handwriting corrections for common misreadings
HANDWRITING_FIXES = {
    "Aey": "Hey",
    "Ehere": "There",
    "l": "I",
    "|": "I",
    "1": "I",
    "ipv6": "I",
    ")": ",",
    "(8": "is"
}

# Add custom words to spell checker
spell.word_frequency.load_words(['redditors', 'cursives', 'reddit', 'subreddit', 'upvote', 'downvote'])

DEMO_USER = {"email":"demo@notex.local","password":"demo123","name":"Demo User"}

# Symbol alternatives for uncertain predictions (common OCR confusions)
SYMBOL_ALTERNATIVES = {
    "0": ["O", "o", "θ", "Θ"],
    "O": ["0", "o", "θ", "Θ"],
    "o": ["0", "O", "°"],
    "1": ["l", "I", "|", "i"],
    "l": ["1", "I", "|", "i"],
    "I": ["1", "l", "|", "i"],
    "x": ["×", "X", "χ", "*"],
    "X": ["×", "x", "χ", "*"],
    "2": ["z", "Z", "²"],
    "z": ["2", "Z"],
    "Z": ["2", "z"],
    "5": ["S", "s", "$"],
    "S": ["5", "$", "s"],
    "8": ["B", "&", "∞"],
    "B": ["8", "β", "b"],
    "n": ["η", "ν", "u"],
    "u": ["μ", "ν", "n"],
    "a": ["α", "∂", "d"],
    "d": ["∂", "δ", "a"],
    "e": ["ε", "∈", "c"],
    "p": ["ρ", "π", "P"],
    "r": ["γ", "τ"],
    "t": ["τ", "+", "T"],
    "w": ["ω", "W"],
    "y": ["γ", "Y"],
    "A": ["Λ", "∆", "△"],
    "E": ["Σ", "∈", "ε"],
    "+": ["t", "†", "±"],
    "-": ["−", "–", "—"],
    "=": ["≈", "≡", "≠"],
    "<": ["≤", "«", "‹"],
    ">": ["≥", "»", "›"],
    "/": ["÷", "\\\\", "|"],
    "*": ["×", "·", "∗"],
    "^": ["∧", "˄"],
    "v": ["∨", "ν", "V"],
    "(": ["[", "{", "〈"],
    ")": ["]", "}", "〉"],
}

# -----------------------------
# History Helpers
# -----------------------------
def load_history():
    """Load history from JSON file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history_list):
    """Save history to JSON file"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_list, f, indent=2, ensure_ascii=False)

def add_history_entry(title, image_filename, latex):
    """Add a new entry to history"""
    history = load_history()
    entry = {
        "id": str(uuid.uuid4()),
        "title": title,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_filename": image_filename,
        "latex": latex
    }
    # Add to beginning (newest first)
    history.insert(0, entry)
    # Keep only last 50 entries
    history = history[:50]
    save_history(history)
    return entry

def preprocess_image_for_ocr(image_path):
    """
    Preprocess image to improve OCR accuracy for handwriting.
    Uses gentle preprocessing to preserve detail.
    Returns path to preprocessed image.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Resize if too small (OCR works better with larger images)
    height, width = gray.shape
    if max(height, width) < 1500:
        scale = 1500 / max(height, width)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 2. Light denoising (preserve edges)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 3. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # 4. Normalize to improve consistency
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    
    # Save preprocessed image (keep as grayscale, NOT binary)
    preprocessed_path = image_path.replace('.', '_preprocessed.')
    cv2.imwrite(preprocessed_path, normalized)
    
    return preprocessed_path

def sort_ocr_results(results):
    """
    Sort OCR results by reading order (top-to-bottom, left-to-right).
    Groups text into lines using dynamic height-based tolerance.
    """
    if not results:
        return []
        
    # Helper to get box geometry
    def get_box_stats(bbox):
        # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        return {
            'min_y': min(ys),
            'max_y': max(ys),
            'center_y': sum(ys)/4,
            'center_x': sum(xs)/4,
            'height': max(ys) - min(ys)
        }

    # Annotate results with stats
    boxes = []
    for item in results:
        stats = get_box_stats(item[0])
        boxes.append({'item': item, 'stats': stats})
    
    # Sort initially by top-Y to process from top down
    boxes.sort(key=lambda b: b['stats']['min_y'])
    
    lines = []
    current_line = []
    
    for box in boxes:
        if not current_line:
            current_line.append(box)
            continue
            
        # Check against average stats of current line
        line_avg_y = sum(b['stats']['center_y'] for b in current_line) / len(current_line)
        line_avg_height = sum(b['stats']['height'] for b in current_line) / len(current_line)
        
        # Determine strictness based on line height
        # If box center is within 50% of line height from line center, it's the same line
        tolerance = line_avg_height * 0.5
        
        if abs(box['stats']['center_y'] - line_avg_y) < tolerance:
            current_line.append(box)
        else:
            # Start new line
            lines.append(current_line)
            current_line = [box]
            
    if current_line:
        lines.append(current_line)
        
    # Sort each line by X-coordinate
    final_lines = []
    for line in lines:
        line.sort(key=lambda b: b['stats']['center_x'])
        final_lines.append([b['item'] for b in line])
        
    return final_lines

@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email", "")
    password = data.get("password", "")
    if email == DEMO_USER["email"] and password == DEMO_USER["password"]:
        return jsonify({"token":"demo-token-123","user":{"name":DEMO_USER["name"], "email":DEMO_USER["email"]}})
    return jsonify({"message":"Invalid credentials"}), 401

@app.route("/convert", methods=["POST"])
def convert():
    file = request.files.get("file")

    if not file:
        return jsonify({"message": "No file uploaded"}), 400

    # -----------------------------
    # Save uploaded image with unique name
    # -----------------------------
    original_filename = secure_filename(file.filename)
    # Add timestamp to filename to avoid overwrites
    name, ext = os.path.splitext(original_filename)
    unique_filename = f"{name}_{int(time.time())}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    file.save(file_path)

    try:
        # -----------------------------
        # Preprocess image for better OCR
        # -----------------------------
        preprocessed_path = preprocess_image_for_ocr(file_path)
        
        # -----------------------------
        # EasyOCR Handwriting Recognition
        # -----------------------------
        # EasyOCR handles multi-line text automatically
        # Use paragraph=False to detect individual words
        # Use EasyOCR for recognition
        # paragraph=False to detect individual words
        # Reduce link_threshold and width_ths to prevent merging of close handwritten words
        results = ocr_reader.readtext(
            preprocessed_path, 
            detail=1, 
            paragraph=False,
            link_threshold=0.1,  # Default 0.4 - Lower value separates text more aggressively
            width_ths=0.1,       # Default 0.5 - Lower value separates close horizontal text
            canvas_size=2560     # Larger canvas for upscaled image
        )
        
        # Sort results by reading order (returns list of lists)
        sorted_lines = sort_ocr_results(results)
        
        # Extract text and build LaTeX lines
        word_data = []
        latex_lines = []
        
        for line in sorted_lines:
            line_parts = []
            for detection in line:
                if len(detection) == 3:
                    bbox, text, confidence = detection
                else:
                    bbox, text = detection
                    confidence = 0.8
                
                text = text.strip()
                if not text:
                    continue
                    
                # Apply custom handwriting fixes
                if text in HANDWRITING_FIXES:
                    text = HANDWRITING_FIXES[text]
                else:
                    if text == "|": text = "I"
                    if text == ")": text = ","
                    if text == "(": text = ","
                    
                # Spell check correction
                clean_text = ''.join(c for c in text if c.isalpha())
                if len(clean_text) >= 3:
                    if not spell.known([clean_text]):
                        corrected = spell.correction(clean_text)
                        if corrected and corrected != clean_text:
                            text = text.replace(clean_text, corrected)
    
                line_parts.append(text)
                
                # word_data collection (same as before)
                is_uncertain = False
                suggestions = []
                if len(clean_text) >= 3:
                    candidates = spell.candidates(clean_text)
                    if candidates:
                        suggestions = list(candidates)[:3]
                
                word_data.append({
                    "text": text,
                    "original": text,
                    "confidence": int(float(confidence) * 100),
                    "isUncertain": is_uncertain,
                    "suggestions": suggestions
                })
            
            if line_parts:
                # Join words with space and wrap in \text for math mode spacing
                line_content = " ".join(line_parts)
                latex_lines.append(f"& \\text{{{line_content}}}")
                
        # Join lines with LaTeX newline
        recognized_text = " \\\\ \n".join(latex_lines)
        
        if not word_data:
            word_data.append({
                "text": "Text could not be reliably detected",
                "original": "",
                "confidence": 0,
                "isUncertain": True,
                "suggestions": []
            })
            recognized_text = "Text could not be reliably detected"

        # Count uncertain words
        uncertain_count = sum(1 for w in word_data if w.get("isUncertain", False))

        # -----------------------------
        # BUILD LaTeX OUTPUT
        # -----------------------------
        # Build LaTeX output
        latex_output = f"""
\\begin{{aligned}}
{recognized_text}
\\end{{aligned}}
"""


        # -----------------------------
        # Save to history
        # -----------------------------
        add_history_entry(
            title=original_filename,
            image_filename=unique_filename,
            latex=latex_output
        )

        # -----------------------------
        # SEND RESPONSE
        # -----------------------------
        return jsonify({
            "latex": latex_output, 
            "image_filename": unique_filename,
            "words": word_data,
            "uncertainCount": uncertain_count,
            "totalWords": len(word_data)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "message": "Error while processing image",
            "error": str(e)
        }), 500


@app.route("/history", methods=["GET"])
def history():
    """Return history with image URLs"""
    history_list = load_history()
    return jsonify({"history": history_list})

@app.route("/uploads/<filename>", methods=["GET"])
def serve_upload(filename):
    """Serve uploaded images"""
    return send_from_directory(UPLOAD_DIR, filename)

# -----------------------------
# Uncertainty Detection Helpers
# -----------------------------
def get_suggestions_for_word(word):
    """Generate alternative suggestions for a word based on common OCR confusions"""
    suggestions = set()
    for i, char in enumerate(word):
        if char in SYMBOL_ALTERNATIVES:
            for alt in SYMBOL_ALTERNATIVES[char][:3]:  # Limit to 3 alternatives per char
                new_word = word[:i] + alt + word[i+1:]
                suggestions.add(new_word)
    return list(suggestions)[:5]  # Return max 5 suggestions

def postprocess_text(text):
    """Apply common math/symbol fixes"""
    replacements = {
        "pi": "\\pi", "x2": "x^2", "x 2": "x^2",
        "sqrt": "\\sqrt{}", "->": "\\rightarrow"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# Confidence thresholds
CONFIDENCE_HIGH = 80    # Above this = confident
CONFIDENCE_LOW = 40     # Below this = rejected
# Between 40-80 = uncertain (included but flagged)

def process_single_image(file):
    """Process a single image and return result dict using EasyOCR"""
    original_filename = secure_filename(file.filename)
    name, ext = os.path.splitext(original_filename)
    unique_filename = f"{name}_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        file.save(file_path)
        
        # Preprocess image for better OCR
        preprocessed_path = preprocess_image_for_ocr(file_path)
        
        # Use EasyOCR for recognition
        # Tuning parameters for handwriting separation
        results = ocr_reader.readtext(
            preprocessed_path, 
            detail=1, 
            paragraph=False,
            link_threshold=0.1,
            width_ths=0.1,
            canvas_size=2560
        )
        
        # Sort results by reading order (returns list of lists)
        sorted_lines = sort_ocr_results(results)
        
        # Extract text
        word_data = []
        latex_lines = []
        
        for line in sorted_lines:
            line_parts = []
            for detection in line:
                if len(detection) == 3:
                    bbox, text, confidence = detection
                else:
                    bbox, text = detection
                    confidence = 0.8
                
                text = text.strip()
                if not text:
                    continue
                
                # Apply custom handwriting fixes
                if text in HANDWRITING_FIXES:
                    text = HANDWRITING_FIXES[text]
                else:
                    if text == "|": text = "I"
                    if text == ")": text = ","
                    if text == "(": text = ","
                
                # Spell check correction
                clean_text = ''.join(c for c in text if c.isalpha())
                if len(clean_text) >= 3:
                    if not spell.known([clean_text]):
                        corrected = spell.correction(clean_text)
                        if corrected and corrected != clean_text:
                            text = text.replace(clean_text, corrected)
                
                line_parts.append(text)
                
                is_uncertain = bool(confidence < 0.7)
                
                # Get spelling suggestions
                suggestions = []
                if len(clean_text) >= 3:
                    candidates = spell.candidates(clean_text)
                    if candidates:
                        suggestions = list(candidates)[:3]
    
                word_data.append({
                    "text": text,
                    "original": text,
                    "confidence": int(float(confidence) * 100),
                    "isUncertain": is_uncertain,
                    "suggestions": suggestions
                })
            
            if line_parts:
                line_content = " ".join(line_parts)
                latex_lines.append(f"& \\text{{{line_content}}}")
                
        # Join lines with LaTeX newline
        recognized_text = " \\\\ \n".join(latex_lines)
        
        if not word_data:
            word_data.append({
                "text": "Text could not be reliably detected",
                "original": "",
                "confidence": 0,
                "isUncertain": True,
                "suggestions": []
            })
            recognized_text = "Text could not be reliably detected"

        uncertain_count = sum(1 for w in word_data if w.get("isUncertain", False))

        # Build LaTeX output
        # Build LaTeX output
        latex_output = f"""
\\begin{{aligned}}
{recognized_text}
\\end{{aligned}}
"""


        # Save to history
        add_history_entry(title=original_filename, image_filename=unique_filename, latex=latex_output)

        return {
            "filename": original_filename,
            "status": "success",
            "latex": latex_output,
            "image_filename": unique_filename,
            "words": word_data,
            "uncertainCount": uncertain_count,
            "totalWords": len(word_data)
        }

    except Exception as e:
        return {
            "filename": original_filename,
            "status": "error",
            "error": str(e)
        }

@app.route("/convert/batch", methods=["POST"])
def convert_batch():
    """Process multiple files at once"""
    files = request.files.getlist("files")
    
    if not files or len(files) == 0:
        return jsonify({"message": "No files uploaded"}), 400
    
    results = []
    for file in files:
        if file and file.filename:
            result = process_single_image(file)
            results.append(result)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    
    return jsonify({
        "results": results,
        "total": len(results),
        "success": success_count,
        "failed": len(results) - success_count
    })

@app.route("/export/tex", methods=["POST"])
def export_tex():
    data = request.get_json() or {}
    latex = data.get("latex","")
    if not latex:
        return jsonify({"message":"No latex provided"}), 400
    buf = io.BytesIO(latex.encode("utf-8"))
    return send_file(buf, download_name="export.tex", as_attachment=True, mimetype="text/x-tex")

@app.route("/export/pdf", methods=["POST"])
def export_pdf():
    data = request.get_json() or {}
    latex = data.get("latex", "")

    if not latex:
        return jsonify({"message": "No LaTeX provided"}), 400

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    story = []

    # Clean LaTeX text for PDF
    # Remove wrappers and LaTeX command syntax to show only the content
    clean_lines = []
    for line in latex.split("\n"):
        line = line.strip()
        if not line: continue
        if "\\begin{aligned}" in line or "\\end{aligned}" in line:
            continue
        
        # Remove & \text{...} \\ wrapper logic
        clean = line.replace("&", "").replace("\\\\", "")
        if "\\text{" in clean:
            clean = clean.replace("\\text{", "")
            # Remove the last closing brace corresponding to \text{
            # Since my generator puts it at the end, this is safe-ish
            if clean.endswith("}"):
                clean = clean[:-1]
        
        clean_lines.append(clean.strip())

    # Create PDF paragraphs
    for line in clean_lines:
        story.append(Paragraph(line, styles["Normal"]))

    doc.build(story)

    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="converted.pdf",
        mimetype="application/pdf"
    )


if __name__ == "__main__":
    app.run(port=5000, debug=True)
