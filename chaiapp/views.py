import cv2
import numpy as np
import base64
import io
import math
import random
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.shortcuts import render

# Constants
TEASPOON_ML = 5.0
GLASS_BOTTOM_DIAM_CM = 4.0
GLASS_TOP_DIAM_CM = 5.0
GLASS_HEIGHT_CM = 10.0
MAX_DIMENSION = 800

def home(request):
    return render(request, "index.html")

def decode_image(data_url):
    """Decode base64 image and resize if needed"""
    header, encoded = data_url.split(",", 1)
    image_data = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
    # Increase sharpness by 50%
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(25)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Resize if too large
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), 
                        interpolation=cv2.INTER_AREA)
    return img

def find_cup_contour(img):
    """Find the cup contour using edge detection and contour analysis"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Morphological closing to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Find the largest contour with reasonable aspect ratio
    best_cnt = None
    best_area = 0
    img_area = img.shape[0] * img.shape[1]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.01:  # Too small
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(h) / w
        
        if 0.8 < aspect_ratio < 3.0 and area > best_area:
            best_area = area
            best_cnt = cnt
    
    return best_cnt

def segment_chai_froth(img, cup_contour):
    """Segment chai and froth layers using color and texture analysis"""
    if cup_contour is None:
        return img.shape[0], 0, img.shape[0], img  # Fallback values
    
    # Create mask for the cup
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cup_contour], -1, 255, -1)
    
    # Get ROI
    x, y, w, h = cv2.boundingRect(cup_contour)
    roi = img[y:y+h, x:x+w]
    roi_mask = mask[y:y+h, x:x+w]
    
    # Convert to LAB color space for better color analysis
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    
    # Analyze lightness row by row
    row_means = np.mean(L, axis=1)
    smoothed = cv2.GaussianBlur(row_means.reshape(-1, 1), (21, 1), 0).flatten()
    
    # Find the froth boundary (where lightness changes most rapidly)
    gradient = np.abs(np.gradient(smoothed))
    froth_bottom = np.argmax(gradient[int(h*0.1):int(h*0.9)]) + int(h*0.1)
    
    # Validate froth detection
    top_mean = np.mean(smoothed[:froth_bottom])
    bottom_mean = np.mean(smoothed[froth_bottom:])
    
    if top_mean - bottom_mean < 5:  # Not enough contrast
        froth_bottom = 0
    
    # Calculate heights
    cup_height = h
    froth_height = froth_bottom
    chai_height = cup_height - froth_height
    
    annotated = roi.copy()
    
    # Draw bounding boxes
    if froth_bottom > 0:
        # Froth box (blue)
        cv2.rectangle(annotated, (0, 0), (w-1, froth_bottom), (255, 0, 0), 2)
        # Chai box (green)
        cv2.rectangle(annotated, (0, froth_bottom), (w-1, h-1), (0, 255, 0), 2)
    
    # Draw cup contour on original image for context
    cv2.drawContours(img, [cup_contour], -1, (255, 0, 0), 2)
    img[y:y+h, x:x+w] = annotated
    
    return chai_height, froth_height, cup_height, img

def frustum_volume(height, bottom_diam, top_diam):
    """Calculate volume of a frustum (tapered cylinder)"""
    if height <= 0:
        return 0.0
    return (math.pi * height / 12.0) * (bottom_diam**2 + bottom_diam*top_diam + top_diam**2)

def calculate_volumes(chai_px, froth_px, cup_px):
    """Calculate volumes based on pixel measurements"""
    if cup_px <= 0:
        return 0.0, 0.0, None, 0.0, 0.0
    
    px_to_cm = GLASS_HEIGHT_CM / cup_px
    chai_cm = chai_px * px_to_cm
    froth_cm = froth_px * px_to_cm
    
    # Total cup volume
    total_ml = frustum_volume(GLASS_HEIGHT_CM, GLASS_BOTTOM_DIAM_CM, GLASS_TOP_DIAM_CM)
    
    # Chai volume (frustum from bottom to liquid level)
    if chai_cm >= GLASS_HEIGHT_CM:
        chai_ml = total_ml
    else:
        # Calculate diameter at liquid level
        D_liquid = GLASS_BOTTOM_DIAM_CM + (GLASS_TOP_DIAM_CM - GLASS_BOTTOM_DIAM_CM) * (chai_cm / GLASS_HEIGHT_CM)
        chai_ml = frustum_volume(chai_cm, GLASS_BOTTOM_DIAM_CM, D_liquid)
    
    # Froth volume is the remaining space
    froth_ml = max(0.0, total_ml - chai_ml)
    
    # Calculate percentages and ratio
    chai_pct = (chai_px / cup_px) * 100
    froth_pct = (froth_px / cup_px) * 100
    ratio = chai_ml / froth_ml if froth_ml > 0 else None
    
    # Teaspoons
    chai_tsp = chai_ml / TEASPOON_ML
    
    return chai_pct, froth_pct, ratio, chai_ml, chai_tsp, froth_ml

def count_bubbles(img):
    """Count bubbles in the froth using Hough Circle Transform"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    
    # Detect circles (bubbles)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=50,
        param2=20,
        minRadius=5,
        maxRadius=30
    )
    
    annotated = img.copy()
    bubble_count = 0
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Filter and count valid circles
        for i in circles[0, :]:
            # Only count circles in the central area (avoid edges)
            if (i[0] > 50 and i[0] < img.shape[1]-50 and 
                i[1] > 50 and i[1] < img.shape[0]-50):
                cv2.circle(annotated, (i[0], i[1]), i[2], (0, 255, 0), 2)
                bubble_count += 1
    
    return bubble_count, annotated

def generate_roast(chai_pct, froth_pct, ratio):
    """Generate a fun roast based on the chai characteristics with random witty responses"""
    roasts = {
    "pure_chai": [
        "This chai is so pure, monks might meditate on it.",
        "No froth at all — this cup means business.",
        "Looks like you skipped the froth department entirely.",
        "Froth? Never heard of her.",
        "As naked as chai gets.",
        "A minimalist masterpiece — just chai, no nonsense.",
        "This is chai stripped down to its soul.",
        "Froth-phobic much?",
        "This chai could pass airport security without a froth check.",
        "The Sahara has more bubbles than this."
    ],
    "empty": [
        "Congratulations, you’ve invented invisible chai.",
        "Air-flavored chai? Bold choice.",
        "I think you just sent me a picture of disappointment.",
        "Was the chai stolen before the photo?",
        "This cup has trust issues — nothing inside.",
        "The emptiness speaks louder than words.",
        "Cup is ready... for literally anything but chai.",
        "So empty, it echoes.",
        "Not a chai, just a ceramic cry for help.",
        "The only thing brewing here is sadness."
    ],
    "chai_heavy": [
        "This chai could knock out a caffeine rookie.",
        "Froth is scared to exist here.",
        "Looks like a chai ocean with a froth island.",
        "Strong enough to wake your ancestors.",
        "You clearly don’t believe in balance.",
        "This chai skipped yoga class.",
        "Liquid dominance achieved.",
        "Your chai is basically flexing right now.",
        "The froth drowned before it could fight.",
        "Chai here, chai there, chai everywhere."
    ],
    "froth_heavy": [
        "This isn’t chai, it’s a bubble bath.",
        "More froth than chai — Starbucks would be proud.",
        "I’ve seen lattes with more chai than this.",
        "You’re one step away from whipped cream.",
        "Chai drowned under a froth tsunami.",
        "Bubble kingdom, chai peasants.",
        "This is a cappuccino cosplay.",
        "Froth domination: 100%.",
        "The chai is playing hide-and-seek under there.",
        "Did you milk the clouds for this?"
    ],
    "balanced": [
        "Balanced like a Zen master’s tea.",
        "This is chai harmony in a cup.",
        "If chai could win beauty contests, this would.",
        "Perfect ratio — a rare sight indeed.",
        "Chai gods are pleased with you.",
        "Michelangelo would paint this cup.",
        "As symmetrical as a monk’s garden.",
        "Your chai has achieved inner peace.",
        "Drink it before the UN declares it a heritage.",
        "Balanced enough to inspire poetry."
    ]
}

    
    if ratio is None and chai_pct > 0:
        return random.choice(roasts["pure_chai"])
    if ratio is None and chai_pct == 0:
        return random.choice(roasts["empty"])
    if ratio is not None and ratio > 10:
        return random.choice(roasts["chai_heavy"])
    if ratio is not None and ratio < 0.5:
        return random.choice(roasts["froth_heavy"])
    return random.choice(roasts["balanced"])

@csrf_exempt
def analyze_chai(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=400)
    
    try:
        data = json.loads(request.body)
        side_view = data.get('side_view')
        top_view = data.get('top_view')
        if not side_view or not top_view:
            return JsonResponse({'error': 'Both side and top views required'}, status=400)
        
        # Process side view for volume estimation
        side_img = decode_image(side_view)
        cup_contour = find_cup_contour(side_img)
        chai_px, froth_px, cup_px, annotated_side = segment_chai_froth(side_img, cup_contour)
        
        # Calculate volumes and metrics
        chai_pct, froth_pct, ratio, chai_ml, chai_tsp, froth_ml = calculate_volumes(
            chai_px, froth_px, cup_px
        )
        
        # Process top view for bubble counting
        top_img = decode_image(top_view)
        bubble_count, annotated_top = count_bubbles(top_img)
        
        # Generate roast comment
        roast = generate_roast(chai_pct, froth_pct, ratio)
        
        # Convert annotated images to base64
        _, side_encoded = cv2.imencode('.jpg', annotated_side)
        side_base64 = base64.b64encode(side_encoded).decode('utf-8')
        
        _, top_encoded = cv2.imencode('.jpg', annotated_top)
        top_base64 = base64.b64encode(top_encoded).decode('utf-8')
        
        # Prepare response
        response = {
            'chai_height': round(chai_pct, 2),
            'froth_height': round(froth_pct, 2),
            'ratio': round(ratio, 2) if ratio is not None else None,
            'chai_ml': round(chai_ml, 2),
            'chai_teaspoons': round(chai_tsp, 2),
            'froth_ml': round(froth_ml, 2),
            'bubble_count': bubble_count,
            'roast': roast,
            'annotated_side': 'data:image/jpeg;base64,' + side_base64,
            'annotated_top': 'data:image/jpeg;base64,' + top_base64
        }
        
        # Log detailed results to console (for developer)
        print("\n=== Chai Analysis Results ===")
        print(f"Chai Height: {response['chai_height']}%")
        print(f"Froth Height: {response['froth_height']}%")
        print(f"Ratio (Chai:Froth): {response['ratio'] or 'N/A'}")
        print(f"Chai Volume: {response['chai_ml']} ml ({response['chai_teaspoons']} tsp)")
        print(f"Froth Volume: {response['froth_ml']} ml")
        print(f"Bubble Count: {response['bubble_count']}")
        print(f"Roast: {response['roast']}")
        
        return JsonResponse(response)
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)
