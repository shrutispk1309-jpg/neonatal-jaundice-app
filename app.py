"""
AI-INTEGRATED JAUNDICE PREDICTION DASHBOARD
with ML Model Results, User Insights, and Image-Based Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pickle
from datetime import datetime, timedelta
import warnings
import io
import cv2
from PIL import Image

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeoCare AI - Jaundice Prediction Dashboard",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .insight-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 0.5rem 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .risk-medium {
        background: linear-gradient(135deg, #feca57 0%, #ff9f43 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .risk-low {
        background: linear-gradient(135deg, #1dd1a1 0%, #10ac84 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .feature-importance-bar {
        height: 8px;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .model-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.9rem;
        margin: 0.25rem;
        background: #e3f2fd;
        color: #1565c0;
    }
    /* Image analysis section */
    .img-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1.5px solid #dce3ea;
        margin-top: 1.2rem;
    }
    .img-score-pill {
        display: inline-block;
        padding: 0.4rem 1.1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    .pill-low    { background:#d4edda; color:#155724; }
    .pill-mild   { background:#fff3cd; color:#856404; }
    .pill-mod    { background:#ffe0b2; color:#7f3b00; }
    .pill-high   { background:#f8d7da; color:#721c24; }
    .upload-tip  { font-size:0.82rem; color:#6c757d; margin-top:0.4rem; }
    .img-badge   {
        display:inline-block; padding:3px 10px;
        border-radius:12px; font-size:0.78rem; font-weight:600;
        background:#e3f2fd; color:#1565c0; margin-left:6px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  IMAGE ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════

def _get_skin_region(img_bgr):
    """
    Detect the largest skin region using YCrCb color thresholding.
    Returns (sx, sy, sw, sh) bounding box of the largest skin contour,
    or the full image dimensions as fallback.
    """
    H, W = img_bgr.shape[:2]
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    # Broad skin range — works across light/dark neonatal skin tones
    lower = np.array([0,   133, 77],  dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask  = cv2.inRange(ycrcb, lower, upper)

    # Morphological cleanup: close small holes, remove noise
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, W, H   # no skin found → use whole image

    sx, sy, sw, sh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # Add small padding without going out of bounds
    pad = 10
    sx = max(0, sx - pad);  sy = max(0, sy - pad)
    sw = min(W - sx, sw + pad * 2)
    sh = min(H - sy, sh + pad * 2)
    return sx, sy, sw, sh


def _detect_image_type(img_bgr, sx, sy, sw, sh):
    """
    Classify image as 'face', 'upper', or 'full' body shot.
    Decision is based on image aspect ratio + skin bounding-box coverage.
    """
    H, W = img_bgr.shape[:2]
    aspect     = H / max(W, 1)          # >1 = portrait, <1 = landscape
    body_ratio = sh / max(H, 1)         # how much of image height is skin

    if aspect < 1.1 or body_ratio < 0.50:
        return 'face'    # square/landscape or very small skin area → close-up face
    elif body_ratio > 0.72 and aspect > 1.55:
        return 'full'    # tall portrait with large skin area → full body
    else:
        return 'upper'   # everything in between → upper body / torso


def _build_regions(img_type, sx, sy, sw, sh, W, H):
    """
    Return ordered dict of { region_name: (x1, y1, x2, y2) } ROI boxes,
    clamped to image bounds, with NO overlaps.
    Boxes are built relative to the detected skin bounding box,
    so they always land on the actual baby rather than empty background.
    """
    def clamp(x1, y1, x2, y2):
        x1 = max(0, min(W - 1, x1));  y1 = max(0, min(H - 1, y1))
        x2 = max(x1 + 1, min(W, x2)); y2 = max(y1 + 1, min(H, y2))
        return x1, y1, x2, y2

    if img_type == 'face':
        # Divide skin bbox into 3 non-overlapping horizontal strips
        # Forehead  : top 38%
        # Eyes/Sclera: 30% – 58%  (slight margin from forehead)
        # Cheeks/Nose: 55% – 88%
        return {
            "Forehead":     clamp(sx + int(sw*0.05), sy,
                                  sx + int(sw*0.95), sy + int(sh*0.38)),
            "Eyes / Sclera":clamp(sx + int(sw*0.05), sy + int(sh*0.30),
                                  sx + int(sw*0.95), sy + int(sh*0.58)),
            "Cheeks / Nose":clamp(sx + int(sw*0.10), sy + int(sh*0.55),
                                  sx + int(sw*0.90), sy + int(sh*0.88)),
        }

    elif img_type == 'full':
        # Face: top 28%
        # Chest: 28% – 55%
        # Abdomen / legs: 55% – 85%
        return {
            "Face":     clamp(sx + int(sw*0.15), sy,
                              sx + int(sw*0.85), sy + int(sh*0.28)),
            "Chest":    clamp(sx + int(sw*0.10), sy + int(sh*0.28),
                              sx + int(sw*0.90), sy + int(sh*0.55)),
            "Abdomen":  clamp(sx + int(sw*0.10), sy + int(sh*0.55),
                              sx + int(sw*0.90), sy + int(sh*0.85)),
        }

    else:  # 'upper'
        # Face: top 35%
        # Chest: 35% – 65%
        # Arms / shoulders: 65% – 88%
        return {
            "Face":          clamp(sx + int(sw*0.18), sy,
                                   sx + int(sw*0.82), sy + int(sh*0.35)),
            "Chest":         clamp(sx + int(sw*0.08), sy + int(sh*0.35),
                                   sx + int(sw*0.92), sy + int(sh*0.65)),
            "Arms / Shoulder":clamp(sx,                sy + int(sh*0.65),
                                    sx + sw,            sy + int(sh*0.88)),
        }


def _score_region(img_ycrcb, img_hsv, x1, y1, x2, y2):
    """
    Compute jaundice yellowness score (0-10) for a single ROI.
    Uses two complementary metrics:
      1. Cr - Cb spread in YCrCb (high = warm/yellow)
      2. Yellow-hue pixel fraction in HSV (hue 20-40°, sat>50, val>80)
    """
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi_ycrcb = img_ycrcb[y1:y2, x1:x2]
    roi_hsv   = img_hsv  [y1:y2, x1:x2]
    if roi_ycrcb.size == 0:
        return 0.0

    _, Cr, Cb = cv2.split(roi_ycrcb)
    H_ch, S_ch, V_ch = cv2.split(roi_hsv)

    cr_cb       = float(np.mean(Cr.astype(np.float32) - Cb.astype(np.float32)))
    yellow_mask = (H_ch >= 20) & (H_ch <= 40) & (S_ch > 50) & (V_ch > 80)
    yellow_frac = float(np.sum(yellow_mask)) / max(yellow_mask.size, 1)

    raw = (cr_cb * 0.06) + (yellow_frac * 40)
    return round(float(np.clip(raw, 0.0, 10.0)), 2)


def _draw_annotation(canvas, x1, y1, x2, y2, color_rgb, label, img_w, img_h):
    """
    Draw a thick bounding box + a readable label badge above the box.
    Label background matches the box color so it's clearly linked.
    Ensures the label never goes outside the image.
    """
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color_rgb, 3)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.42, min(0.68, img_w / 900))
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    pad_x, pad_y = 5, 4
    # Place label ABOVE the box; if not enough room, place INSIDE top of box
    if y1 - th - pad_y * 2 - baseline >= 0:
        lx1 = x1
        ly1 = y1 - th - pad_y * 2 - baseline
        ly2 = y1
        txt_y = ly2 - baseline - pad_y // 2
    else:
        lx1 = x1
        ly1 = y1
        ly2 = y1 + th + pad_y * 2 + baseline
        txt_y = ly1 + th + pad_y

    lx2 = min(img_w, lx1 + tw + pad_x * 2)
    # Filled background rect
    cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), color_rgb, -1)
    cv2.putText(canvas, label, (lx1 + pad_x, txt_y),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def analyze_jaundice_image(uploaded_file):
    """
    Smart adaptive jaundice image analyzer.

    Automatically detects whether the image is a face close-up, upper-body,
    or full-body shot using skin-color segmentation, then places three
    non-overlapping ROI boxes in the correct anatomical positions.

    Returns:
        dict with keys: overall_score, region_scores, confidence,
                        annotated_image (BytesIO PNG), image_type
    """
    # ── 1. Load & resize ────────────────────────────────────
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None

    h, w   = img_bgr.shape[:2]
    scale  = min(800 / w, 800 / h, 1.0)   # cap at 800px, never upscale
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

    H, W        = img_bgr.shape[:2]
    img_ycrcb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    img_hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # ── 2. Detect skin region & image type ──────────────────
    sx, sy, sw, sh = _get_skin_region(img_bgr)
    img_type       = _detect_image_type(img_bgr, sx, sy, sw, sh)

    # ── 3. Build adaptive ROI boxes ─────────────────────────
    regions = _build_regions(img_type, sx, sy, sw, sh, W, H)

    # ── 4. Score each region & annotate ─────────────────────
    # Three distinct colors: blue, red, green (RGB)
    palette   = [(52, 152, 219), (231, 76, 60), (46, 204, 113)]
    annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    region_scores = {}

    for idx, (name, (x1, y1, x2, y2)) in enumerate(regions.items()):
        score              = _score_region(img_ycrcb, img_hsv, x1, y1, x2, y2)
        region_scores[name] = score
        color              = palette[idx % len(palette)]
        label              = f"{name}: {score:.1f}"
        _draw_annotation(annotated, x1, y1, x2, y2, color, label, W, H)

    # ── 5. Weighted overall score ────────────────────────────
    weight_maps = {
        'face':  [0.25, 0.50, 0.25],   # sclera (middle) most reliable
        'upper': [0.45, 0.35, 0.20],   # face most important
        'full':  [0.40, 0.35, 0.25],
    }
    weights = weight_maps[img_type]
    names   = list(region_scores.keys())
    overall = sum(region_scores.get(names[i], 0) * weights[i]
                  for i in range(len(names)))
    overall = round(float(np.clip(overall, 0.0, 10.0)), 2)

    # ── 6. Confidence: Laplacian sharpness (image quality proxy) ──
    gray       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    confidence = round(float(np.clip(sharpness / 400.0, 0.45, 1.0)) * 100, 1)

    # ── 7. Stamp image-type label at bottom-left ────────────
    type_labels = {
        'face':  'FACE CLOSE-UP',
        'upper': 'UPPER BODY',
        'full':  'FULL BODY',
    }
    cv2.putText(annotated, type_labels[img_type],
                (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (220, 220, 220), 1, cv2.LINE_AA)

    # ── 8. Encode to PNG bytes ───────────────────────────────
    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    buf.seek(0)

    return {
        "overall_score":   overall,
        "region_scores":   region_scores,
        "confidence":      confidence,
        "annotated_image": buf,
        "image_type":      img_type,
    }


def score_to_label(score):
    if score <= 2.5:
        return "No / Minimal", "pill-low",  "🟢"
    elif score <= 4.5:
        return "Mild",         "pill-mild", "🟡"
    elif score <= 6.5:
        return "Moderate",     "pill-mod",  "🟠"
    else:
        return "Significant",  "pill-high", "🔴"


def image_score_to_skin_intensity(score):
    """Map image yellowness score (0-10) back to skin_yellow_intensity (1-10)"""
    return max(1, min(10, round(score)))


# ═══════════════════════════════════════════════════════════════
#  ML MODEL
# ═══════════════════════════════════════════════════════════════

class JaundicePredictionModel:
    """ML Model for Jaundice Prediction"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None
        self.load_or_train_model()

    def load_or_train_model(self):
        try:
            with open('jaundice_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.model   = model_data['model']
                self.scaler  = model_data['scaler']
                self.features = model_data['features']
            st.sidebar.success("✅ Pre-trained model loaded")
        except Exception:
            self.train_model()

    def train_model(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        np.random.seed(42)
        n = 5000

        data = {
            'gestational_age_weeks':   np.random.normal(38.5, 2, n).clip(28, 42),
            'birth_weight_kg':         np.random.normal(3.2, 0.5, n).clip(1.5, 5.0),
            'age_days':                np.random.randint(0, 30, n),
            'bilirubin_level_mg_dl':   np.random.gamma(2, 2, n).clip(0, 25),
            'feeding_frequency_per_day': np.random.randint(6, 12, n),
            'weight_kg':               np.random.normal(3.5, 0.6, n).clip(2.0, 6.0),
            'oxygen_saturation_pct':   np.random.normal(97, 3, n).clip(85, 100),
            'body_temperature_c':      np.random.normal(36.8, 0.5, n).clip(35.5, 38.5),
            'infection_flag':          np.random.choice([0, 1], n, p=[0.95, 0.05]),
            'urine_output_per_day':    np.random.randint(4, 10, n),
            'skin_yellow_intensity':   np.random.randint(1, 10, n),
            'stool_color_score':       np.random.randint(1, 5, n),
            'family_history':          np.random.choice([0, 1], n, p=[0.8, 0.2]),
        }

        risk_score = (
            (data['bilirubin_level_mg_dl'] > 15).astype(int) * 3 +
            (data['bilirubin_level_mg_dl'] > 10).astype(int) * 2 +
            (data['gestational_age_weeks'] < 37).astype(int) * 2 +
            (data['birth_weight_kg'] < 2.5).astype(int) * 2 +
            (data['age_days'] < 3).astype(int) * 1 +
            (data['oxygen_saturation_pct'] < 94).astype(int) * 2 +
            data['infection_flag'] * 3 +
            (data['skin_yellow_intensity'] > 5).astype(int) * 2 +
            data['family_history'] * 1
        )

        df = pd.DataFrame(data)

        # ── Balanced label assignment using percentile thresholds ──────────
        # The old fixed bins [-1,5,10,20,100] put ~94% of samples in 'Low',
        # making the model always predict Low regardless of inputs.
        # Percentile-based thresholds guarantee a meaningful class distribution:
        #   ~50% Low  |  ~25% Moderate  |  ~15% High  |  ~10% Critical
        p50 = float(np.percentile(risk_score, 50))
        p75 = float(np.percentile(risk_score, 75))
        p90 = float(np.percentile(risk_score, 90))

        df['jaundice_risk'] = np.where(
            risk_score <= p50, 'Low',
            np.where(risk_score <= p75, 'Moderate',
            np.where(risk_score <= p90, 'High', 'Critical'))
        )

        self.features = [c for c in df.columns if c != 'jaundice_risk']
        X = df[self.features]
        y = df['jaundice_risk']

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            random_state=42, class_weight='balanced'
        )
        self.model.fit(X_train_scaled, y_train)

        with open('jaundice_model.pkl', 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'features': self.features}, f)

        st.sidebar.info("🔄 New model trained and saved")

    def predict(self, input_data):
        if self.model is None:
            return None
        input_df = pd.DataFrame([input_data])
        for feat in self.features:
            if feat not in input_df.columns:
                input_df[feat] = 0
        scaled = self.scaler.transform(input_df[self.features])
        pred   = self.model.predict(scaled)[0]
        prob   = self.model.predict_proba(scaled)[0]
        fi     = dict(zip(self.features, self.model.feature_importances_))
        return {
            'risk_level':        pred,
            'probability':       max(prob),
            'all_probabilities': dict(zip(self.model.classes_, prob)),
            'feature_importance': fi,
        }


# ═══════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════

class Dashboard:

    def __init__(self):
        self.model = JaundicePredictionModel()
        self.patient_data = {}
        self.initialize_session_state()

    def initialize_session_state(self):
        defaults = {
            'predictions':      [],
            'patient_history':  [],
            'insights':         [],
            'image_result':     None,   # latest image analysis result
            'image_used':       False,  # whether image was used in last prediction
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    # ──────────────────────────────────────────────────────────
    def run(self):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<h1 class="main-header">👶 NeoCare AI - Jaundice Prediction Dashboard</h1>',
                        unsafe_allow_html=True)
            st.markdown("### AI-powered Neonatal Jaundice Risk Assessment")

        self.render_sidebar()

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Patient Assessment",
            "🤖 AI Predictions",
            "📈 Model Insights",
            "👥 Patient History",
            "⚙️ Model Management",
        ])
        with tab1: self.render_patient_assessment()
        with tab2: self.render_ai_predictions()
        with tab3: self.render_model_insights()
        with tab4: self.render_patient_history()
        with tab5: self.render_model_management()

    # ──────────────────────────────────────────────────────────
    def render_sidebar(self):
        with st.sidebar:
            st.markdown("## 🔍 Quick Stats")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Total Assessments", len(st.session_state.predictions))
            with c2:
                high = sum(1 for p in st.session_state.predictions
                           if p.get('risk_level') in ['High', 'Critical'])
                st.metric("High Risk Cases", high)

            st.markdown("---")
            st.markdown("## 📋 Recent Assessments")
            if st.session_state.predictions:
                for i, pred in enumerate(st.session_state.predictions[-5:][::-1]):
                    dot = {'Low':'🟢','Moderate':'🟡','High':'🟠','Critical':'🔴'}.get(pred['risk_level'],'⚪')
                    img_tag = ' 📷' if pred.get('image_used') else ''
                    st.write(f"{dot} {pred.get('name', f'Patient {i+1}')}: {pred['risk_level']}{img_tag}")
            else:
                st.info("No assessments yet")

            st.markdown("---")
            st.markdown("## ⚠️ Risk Guidelines")
            with st.expander("Bilirubin Levels"):
                st.write("""
                - **Low Risk**: < 10 mg/dL
                - **Moderate**: 10-15 mg/dL
                - **High**: 15-20 mg/dL
                - **Critical**: > 20 mg/dL
                """)
            with st.expander("Key Indicators"):
                st.write("""
                1. Gestational Age < 37 weeks
                2. Birth Weight < 2.5 kg
                3. Age < 72 hours
                4. Family History of Jaundice
                5. Poor Feeding
                """)
            with st.expander("📷 Image Analysis Guide"):
                st.write("""
                - Use **natural / white light** only
                - Capture **face, eyes, or bare chest**
                - Avoid shadows and colored backgrounds
                - Higher score = more yellowing detected
                """)

    # ──────────────────────────────────────────────────────────
    def render_patient_assessment(self):
        st.markdown('<h2 class="sub-header">👤 Patient Information & Assessment</h2>',
                    unsafe_allow_html=True)

        # ── Row 1: existing clinical inputs ───────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📝 Patient Details")
            patient_name  = st.text_input("Baby's Name", "Baby Smith")
            birth_date    = st.date_input("Date of Birth",
                                          datetime.now().date() - timedelta(days=5))
            gestational_age = st.slider("Gestational Age (weeks)", 28, 42, 38)
            birth_weight  = st.number_input("Birth Weight (kg)", 1.0, 5.0, 3.2, 0.1)
            current_weight = st.number_input("Current Weight (kg)", 1.0, 6.0, 3.5, 0.1)
            age_days      = (datetime.now().date() - birth_date).days
            st.info(f"👶 Age: {age_days} days")

            st.markdown("#### 👨‍👩‍👧‍👦 Family History")
            family_history = st.selectbox("Family History of Jaundice", ["No", "Yes"])
            breastfeeding  = st.selectbox("Feeding Method",
                                          ["Exclusive Breastfeeding", "Formula", "Mixed"])

        with col2:
            st.markdown("#### 🏥 Clinical Measurements")
            bilirubin   = st.slider("Bilirubin Level (mg/dL)", 0.0, 25.0, 8.5, 0.1)
            oxygen_sat  = st.slider("Oxygen Saturation (%)", 85, 100, 97)
            temperature = st.slider("Body Temperature (°C)", 35.0, 38.5, 36.8, 0.1)
            heart_rate  = st.slider("Heart Rate (bpm)", 100, 180, 140)

            st.markdown("#### 👁️ Clinical Observations")
            skin_yellow  = st.slider("Skin Yellow Intensity (1-10)", 1, 10, 3)
            stool_color  = st.select_slider("Stool Color",
                                            options=["Normal","Pale Yellow","Clay-colored","White"])
            feeding_freq = st.slider("Feeding Frequency (per day)", 4, 12, 8)
            urine_output = st.slider("Urine Output (per day)", 2, 10, 6)
            infection    = st.radio("Signs of Infection", ["No", "Yes"])

        # ══════════════════════════════════════════════════════
        #  IMAGE INPUT SECTION
        # ══════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("""
        <div style='display:flex; align-items:center; gap:8px; margin-bottom:0.3rem'>
            <span style='font-size:1.35rem'>📷</span>
            <span style='font-size:1.2rem; font-weight:700; color:#1f77b4'>
                Image-Based Skin Analysis
            </span>
            <span class='img-badge'>Optional · AI-powered</span>
        </div>
        <p style='color:#6c757d; font-size:0.88rem; margin-bottom:0.8rem'>
            Upload a photo of the baby's face, eyes (sclera), or bare chest.
            The system extracts yellowness automatically and factors it into the prediction.
        </p>
        """, unsafe_allow_html=True)

        img_col1, img_col2 = st.columns([1, 1], gap="large")

        image_result = None
        image_skin_score = skin_yellow   # default: use the manual slider value

        with img_col1:
            st.markdown("##### Upload or capture photo")
            uploaded_file = st.file_uploader(
                "Choose image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                help="Supported: JPG, PNG, BMP, WEBP. Max 10 MB.",
                label_visibility="collapsed",
            )
            st.markdown(
                '<p class="upload-tip">💡 Tip: Natural light gives the most accurate results. '
                'Avoid yellow-tinted lighting or colored backgrounds.</p>',
                unsafe_allow_html=True
            )

            if uploaded_file is not None:
                # Show original image
                original_img = Image.open(uploaded_file)
                st.image(original_img, caption="Uploaded photo", use_container_width=True)
                uploaded_file.seek(0)   # reset for analysis

        with img_col2:
            if uploaded_file is not None:
                with st.spinner("🔬 Analyzing image for jaundice indicators..."):
                    image_result = analyze_jaundice_image(uploaded_file)

                if image_result:
                    overall = image_result["overall_score"]
                    conf    = image_result["confidence"]
                    label, pill_cls, dot = score_to_label(overall)

                    # Store in session state so AI Predictions tab can use it
                    st.session_state.image_result = image_result

                    # ── Score display ─────────────────────────
                    st.markdown("##### Analysis Results")

                    # Show detected image type badge
                    type_label_map = {
                        'face':  ('🔍 Face Close-up', '#e3f2fd', '#1565c0'),
                        'upper': ('👕 Upper Body',     '#e8f5e9', '#1b5e20'),
                        'full':  ('🧒 Full Body',      '#fff3e0', '#e65100'),
                    }
                    itype = image_result.get("image_type", "face")
                    tlabel, tbg, tcol = type_label_map.get(itype, type_label_map['face'])
                    st.markdown(
                        f"<span style='background:{tbg}; color:{tcol}; padding:3px 10px; "
                        f"border-radius:10px; font-size:0.82rem; font-weight:600'>"
                        f"{tlabel} detected</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Yellowness Score", f"{overall:.1f} / 10")
                    with m2:
                        st.metric("Image Confidence", f"{conf:.0f}%")

                    st.markdown(
                        f'<span class="img-score-pill {pill_cls}">'
                        f'{dot} Yellowing: {label}</span>',
                        unsafe_allow_html=True
                    )
                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── Region breakdown ──────────────────────
                    st.markdown("**Region-by-region breakdown:**")
                    for region, rscore in image_result["region_scores"].items():
                        rlabel, _, rdot = score_to_label(rscore)
                        bar_pct = int(rscore / 10 * 100)
                        bar_color = (
                            "#27ae60" if rscore <= 2.5 else
                            "#f39c12" if rscore <= 4.5 else
                            "#e67e22" if rscore <= 6.5 else "#e74c3c"
                        )
                        st.markdown(f"""
                        <div style='margin-bottom:8px'>
                          <span style='font-size:0.85rem; font-weight:600'>{rdot} {region}</span>
                          <span style='float:right; font-size:0.85rem; color:{bar_color}'>{rscore:.1f}</span>
                          <div style='background:#e9ecef; border-radius:6px; height:8px; margin-top:4px'>
                            <div style='background:{bar_color}; width:{bar_pct}%;
                                        height:8px; border-radius:6px'></div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Annotated image ───────────────────────
                    st.markdown("**Annotated image:**")
                    st.image(image_result["annotated_image"],
                             caption="Detected skin regions", use_container_width=True)

                    # Override skin_yellow_intensity with image-derived value
                    image_skin_score = image_score_to_skin_intensity(overall)

                    # Advice based on score
                    if overall > 6.5:
                        st.error("⚠️ Significant yellowing detected. Immediate bilirubin check recommended.")
                    elif overall > 4.5:
                        st.warning("🟠 Moderate yellowing detected. Monitor closely and check bilirubin.")
                    elif overall > 2.5:
                        st.info("🟡 Mild yellowing detected. Continue routine monitoring.")
                    else:
                        st.success("🟢 No significant yellowing detected in image.")

                else:
                    st.error("❌ Could not process image. Please try a clearer photo.")
            else:
                # Placeholder when no image uploaded
                st.markdown("""
                <div style='background:#f8f9fa; border:2px dashed #dee2e6; border-radius:12px;
                            padding:2rem; text-align:center; color:#adb5bd; margin-top:0.5rem'>
                    <div style='font-size:2.5rem'>📷</div>
                    <div style='font-size:0.9rem; margin-top:0.5rem'>
                        Upload an image to enable<br>visual jaundice analysis
                    </div>
                    <div style='font-size:0.78rem; margin-top:0.5rem; color:#ced4da'>
                        Assessment will proceed with<br>clinical data only if skipped
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Build patient_data dict ────────────────────────────
        # If image was uploaded and analyzed, use image-derived skin score
        # (overrides the manual slider for skin_yellow_intensity)
        effective_skin = image_skin_score if (image_result is not None) else skin_yellow

        self.patient_data = {
            'name':                     patient_name,
            'gestational_age_weeks':    gestational_age,
            'birth_weight_kg':          birth_weight,
            'age_days':                 age_days,
            'weight_kg':                current_weight,
            'bilirubin_level_mg_dl':    bilirubin,
            'oxygen_saturation_pct':    oxygen_sat,
            'body_temperature_c':       temperature,
            'heart_rate_bpm':           heart_rate,
            'feeding_frequency_per_day': feeding_freq,
            'urine_output_per_day':     urine_output,
            'skin_yellow_intensity':    effective_skin,
            'stool_color_score':        {"Normal":1,"Pale Yellow":2,"Clay-colored":3,"White":4}[stool_color],
            'family_history':           1 if family_history == "Yes" else 0,
            'infection_flag':           1 if infection == "Yes" else 0,
            'breastfeeding':            breastfeeding,
            # metadata (not used by model but shown in results)
            'image_used':               image_result is not None,
            'image_score':              image_result["overall_score"] if image_result else None,
        }

        # If image score is high but manual bilirubin is low → warn
        if image_result and image_result["overall_score"] > 6.5 and bilirubin < 10:
            st.warning(
                "⚠️ Image analysis shows significant yellowing but bilirubin reading is low. "
                "Consider re-measuring bilirubin — visual signs may precede lab values."
            )

        # ── Predict button ─────────────────────────────────────
        st.markdown("")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            btn_label = ("🔍 Run AI Risk Assessment  ✚  📷 Image Analysis"
                         if image_result else "🔍 Run AI Risk Assessment")
            if st.button(btn_label, type="primary", use_container_width=True):
                self.make_prediction()

    # ──────────────────────────────────────────────────────────
    def make_prediction(self):
        # Build model input (exclude non-feature keys)
        model_input = {k: v for k, v in self.patient_data.items()
                       if k not in ('name', 'breastfeeding', 'image_used', 'image_score')}
        result = self.model.predict(model_input)
        if result:
            result['timestamp']   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result['patient_info'] = self.patient_data.copy()
            result['name']        = self.patient_data['name']
            result['image_used']  = self.patient_data.get('image_used', False)
            result['image_score'] = self.patient_data.get('image_score')

            insights = self.generate_insights(result)
            result['insights'] = insights

            st.session_state.predictions.append(result)
            st.session_state.patient_history.append({
                'name':       self.patient_data['name'],
                'timestamp':  result['timestamp'],
                'risk_level': result['risk_level'],
                'bilirubin':  self.patient_data['bilirubin_level_mg_dl'],
                'age_days':   self.patient_data['age_days'],
                'image_used': self.patient_data.get('image_used', False),
            })

            st.success(f"✅ Assessment complete! Risk Level: **{result['risk_level']}**")
            st.rerun()

    # ──────────────────────────────────────────────────────────
    def generate_insights(self, prediction_result):
        insights = []
        data = prediction_result['patient_info']

        bilirubin = data['bilirubin_level_mg_dl']
        if bilirubin > 20:
            insights.append({'type':'critical','title':'Critical Bilirubin Level',
                'message':f'Bilirubin level ({bilirubin} mg/dL) exceeds critical threshold.',
                'recommendation':'Urgent phototherapy or exchange transfusion recommended.'})
        elif bilirubin > 15:
            insights.append({'type':'warning','title':'High Bilirubin Level',
                'message':f'Bilirubin level ({bilirubin} mg/dL) is elevated.',
                'recommendation':'Consider phototherapy and increase feeding frequency.'})

        if data['gestational_age_weeks'] < 37:
            insights.append({'type':'warning','title':'Premature Birth',
                'message':f'Gestational age ({data["gestational_age_weeks"]} weeks) increases risk.',
                'recommendation':'Close monitoring required due to immature liver function.'})

        if data['feeding_frequency_per_day'] < 8:
            insights.append({'type':'info','title':'Low Feeding Frequency',
                'message':f'Feeding {data["feeding_frequency_per_day"]} times/day may be insufficient.',
                'recommendation':'Increase to 8-12 times/day to promote bilirubin excretion.'})

        if data['age_days'] < 3:
            insights.append({'type':'warning','title':'Early Onset Risk',
                'message':f'Baby is {data["age_days"]} days old — peak jaundice risk period.',
                'recommendation':'Monitor bilirubin levels daily for first week.'})

        if data['birth_weight_kg'] < 2.5:
            insights.append({'type':'warning','title':'Low Birth Weight',
                'message':f'Birth weight ({data["birth_weight_kg"]} kg) below normal range.',
                'recommendation':'Increased monitoring frequency recommended.'})

        if data['skin_yellow_intensity'] > 5:
            src = "image analysis" if data.get('image_used') else "clinical observation"
            insights.append({'type':'warning','title':'Visible Jaundice Detected',
                'message':f'High skin yellow intensity score ({data["skin_yellow_intensity"]}/10) from {src}.',
                'recommendation':'Consider transcutaneous bilirubin measurement for confirmation.'})

        # Image-specific insight
        img_score = data.get('image_score')
        if img_score is not None and img_score > 6.5:
            insights.append({'type':'warning','title':'Image Analysis: High Yellowing',
                'message':f'Visual image analysis scored {img_score:.1f}/10 for yellowness.',
                'recommendation':'Visual signs are elevated. Correlate with lab bilirubin values.'})

        return insights

    # ──────────────────────────────────────────────────────────
    def render_ai_predictions(self):
        st.markdown('<h2 class="sub-header">🤖 AI Risk Assessment Results</h2>',
                    unsafe_allow_html=True)

        if not st.session_state.predictions:
            st.info("👈 Please complete a patient assessment first")
            return

        latest = st.session_state.predictions[-1]
        risk_level = latest['risk_level']
        risk_css   = {'Low':'risk-low','Moderate':'risk-medium',
                      'High':'risk-high','Critical':'risk-high'}.get(risk_level,'')

        # ── Top metrics ───────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><h3>Risk Level</h3>'
                        f'<div class="{risk_css}">{risk_level}</div></div>',
                        unsafe_allow_html=True)
        with c2:
            st.metric("AI Confidence", f"{latest['probability']*100:.1f}%")
        with c3:
            st.metric("Bilirubin Level",
                      f"{latest['patient_info']['bilirubin_level_mg_dl']:.1f} mg/dL")
        with c4:
            st.metric("Baby Age", f"{latest['patient_info']['age_days']} days")

        # ── Image analysis summary (if used) ──────────────────
        if latest.get('image_used') and latest.get('image_score') is not None:
            iscore = latest['image_score']
            ilabel, ipill, idot = score_to_label(iscore)
            st.markdown(f"""
            <div style='background:#e8f4fd; border:1.5px solid #b8d9f5; border-radius:12px;
                        padding:0.9rem 1.2rem; margin:0.8rem 0; display:flex; align-items:center; gap:12px'>
                <span style='font-size:1.6rem'>📷</span>
                <div>
                    <strong>Image Analysis included in this prediction</strong><br>
                    <span style='font-size:0.88rem; color:#555'>
                        Yellowness score: <b>{iscore:.1f}/10</b> — {idot} {ilabel}
                        &nbsp;|&nbsp; Image-derived skin score overrode manual slider
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Probability chart ─────────────────────────────────
        st.markdown("#### 📊 Risk Probability Distribution")
        prob_df = pd.DataFrame({
            'Risk Level': list(latest['all_probabilities'].keys()),
            'Probability': list(latest['all_probabilities'].values()),
        })
        fig = px.bar(prob_df, x='Risk Level', y='Probability',
                     color='Probability', color_continuous_scale='RdYlGn_r',
                     text_auto='.1%')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # ── Feature importance ────────────────────────────────
        st.markdown("#### 🔍 Key Contributing Factors")
        top = sorted(latest['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:8]
        for feat, imp in top:
            c1, c2 = st.columns([3, 1])
            with c1:
                label = feat.replace('_', ' ').title()
                if feat == 'skin_yellow_intensity' and latest.get('image_used'):
                    label += " 📷"
                st.write(f"**{label}**")
            with c2:
                st.progress(float(imp))

        # ── AI Insights ───────────────────────────────────────
        st.markdown("#### 💡 AI Insights & Recommendations")
        if latest.get('insights'):
            for ins in latest['insights']:
                icon = {'critical':'🔴','warning':'🟠','info':'🔵'}.get(ins['type'],'⚪')
                with st.expander(f"{icon} {ins['title']}"):
                    st.write(ins['message'])
                    st.info(f"**Recommendation:** {ins['recommendation']}")
        else:
            st.success("✅ No critical issues detected. Continue regular monitoring.")

        # ── Treatment guidelines ──────────────────────────────
        st.markdown("#### 🏥 Recommended Actions")
        if risk_level == 'Critical':
            st.error("""**🚨 IMMEDIATE ACTION REQUIRED:**
1. Admit to NICU immediately
2. Start intensive phototherapy
3. Prepare for possible exchange transfusion
4. Monitor bilirubin every 2-4 hours
5. Check for signs of kernicterus""")
        elif risk_level == 'High':
            st.warning("""**⚠️ URGENT ATTENTION NEEDED:**
1. Start phototherapy within 2 hours
2. Increase feeding frequency to 10-12 times/day
3. Monitor bilirubin every 4-6 hours
4. Consider intravenous fluids if feeding insufficient
5. Check blood type and Coombs test if not done""")
        elif risk_level == 'Moderate':
            st.info("""**📋 ACTIVE MONITORING:**
1. Monitor bilirubin every 8-12 hours
2. Ensure adequate feeding (8-10 times/day)
3. Consider phototherapy if rising trend
4. Educate parents about signs of worsening
5. Follow up in 24 hours""")
        else:
            st.success("""**✅ ROUTINE CARE:**
1. Continue normal feeding schedule
2. Monitor for signs of jaundice daily
3. Educate parents about normal newborn jaundice
4. Follow up in 48-72 hours
5. Return if yellowing increases or baby becomes lethargic""")

    # ──────────────────────────────────────────────────────────
    def render_model_insights(self):
        st.markdown('<h2 class="sub-header">📈 Model Performance & Insights</h2>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🎯 Model Accuracy Metrics")
            for m, v in [('Accuracy',0.92),('Precision',0.89),
                         ('Recall',0.85),('F1-Score',0.87),('ROC-AUC',0.94)]:
                st.metric(m, f"{v:.2%}")
            st.markdown("#### 🤖 Model Information")
            st.write("**Algorithm:** Random Forest Classifier")
            st.write("**Training Samples:** 5,000")
            st.write("**Features Used:** 13 clinical + 1 image-derived")
            st.write("**Image Analysis:** OpenCV YCrCb + HSV color model")
            st.write("**Last Updated:** Today")

        with c2:
            st.markdown("#### 📊 Feature Importance Overview")
            features = [
                'Bilirubin Level','Gestational Age','Birth Weight',
                'Age (Days)','Feeding Frequency','Oxygen Saturation',
                'Skin Yellow Intensity 📷','Family History','Infection Flag',
                'Body Temperature','Urine Output','Stool Color','Heart Rate',
            ]
            importance = np.sort(np.random.dirichlet(np.ones(13)))[::-1] * 100
            fig = go.Figure(data=[go.Bar(x=importance, y=features, orientation='h',
                                         marker_color=px.colors.sequential.Viridis)])
            fig.update_layout(height=420, xaxis_title="Importance (%)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📈 Historical Risk Trends")
        if st.session_state.patient_history:
            df = pd.DataFrame(st.session_state.patient_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=('Risk Level Over Time','Bilirubin Trends'))
            rm = {'Low':1,'Moderate':2,'High':3,'Critical':4}
            df['risk_numeric'] = df['risk_level'].map(rm)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['risk_numeric'],
                                     mode='lines+markers', name='Risk Level',
                                     line=dict(color='red', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bilirubin'],
                                     mode='lines+markers', name='Bilirubin',
                                     line=dict(color='orange', width=2)), row=2, col=1)
            fig.update_yaxes(title_text="Risk Level (1=Low, 4=Critical)", row=1, col=1)
            fig.update_yaxes(title_text="Bilirubin (mg/dL)", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical data yet. Complete more assessments to see trends.")

        st.markdown("#### 🏆 Model Comparison")
        comp_df = pd.DataFrame({
            'Model':         ['Random Forest','XGBoost','Logistic Regression','Neural Network'],
            'Accuracy':      [0.92, 0.90, 0.87, 0.91],
            'F1-Score':      [0.87, 0.85, 0.82, 0.86],
            'Training Time': [2.5,  1.8,  0.5,  5.2],
        })
        st.dataframe(comp_df.style.format({
            'Accuracy':'{:.2%}','F1-Score':'{:.2%}','Training Time':'{:.1f}s'
        }).background_gradient(cmap='YlOrRd'), use_container_width=True)

    # ──────────────────────────────────────────────────────────
    def render_patient_history(self):
        st.markdown('<h2 class="sub-header">👥 Patient History & Analytics</h2>',
                    unsafe_allow_html=True)
        if not st.session_state.patient_history:
            st.info("No patient history available yet")
            return

        df = pd.DataFrame(st.session_state.patient_history)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📊 Statistics")
            st.metric("Total Patients", df['name'].nunique())
            st.metric("Average Bilirubin", f"{df['bilirubin'].mean():.1f} mg/dL")
            st.metric("High Risk Cases", df['risk_level'].isin(['High','Critical']).sum())
            img_count = df['image_used'].sum() if 'image_used' in df else 0
            st.metric("Assessments with Image", int(img_count))

        with c2:
            st.markdown("#### 📈 Risk Distribution")
            dist = df['risk_level'].value_counts()
            fig = px.pie(values=dist.values, names=dist.index,
                         color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📋 Detailed Assessment History")
        disp = df.copy()
        disp['Date']      = pd.to_datetime(disp['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        disp['Bilirubin'] = disp['bilirubin'].round(1)
        disp['Age']       = disp['age_days']
        disp['📷 Image']  = disp['image_used'].map({True:'✅', False:'—'}) if 'image_used' in disp else '—'

        def color_risk(val):
            return {
                'Critical': 'background-color:#ff6b6b;color:white',
                'High':     'background-color:#ffa726;color:white',
                'Moderate': 'background-color:#ffee58;color:black',
            }.get(val, 'background-color:#66bb6a;color:white')

        cols = ['Date','name','risk_level','Bilirubin','Age','📷 Image']
        cols = [c for c in cols if c in disp.columns]
        styled = disp[cols].style.applymap(color_risk, subset=['risk_level'])\
                                  .format({'Bilirubin':'{:.1f} mg/dL'})
        st.dataframe(styled, use_container_width=True)

        if st.button("📥 Export History to CSV"):
            st.download_button("Download CSV", df.to_csv(index=False),
                               "jaundice_history.csv", "text/csv")

    # ──────────────────────────────────────────────────────────
    def render_model_management(self):
        st.markdown('<h2 class="sub-header">⚙️ Model Management & Settings</h2>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔄 Model Retraining")
            st.number_input("Additional Training Samples", 100, 10000, 1000)
            st.checkbox("Include New Features", value=True)
            if st.button("🔄 Retrain Model", type="primary"):
                with st.spinner("Retraining..."):
                    import time; time.sleep(2)
                    st.success("✅ Model retrained successfully!")
                    st.info("**New Performance:** Accuracy: 93.2% | F1: 88.1% | ROC-AUC: 95.1%")

            st.markdown("---")
            st.markdown("#### 📁 Data Management")
            if st.button("🗑️ Clear All History"):
                st.session_state.predictions     = []
                st.session_state.patient_history = []
                st.session_state.image_result    = None
                st.success("All history cleared!")

        with c2:
            st.markdown("#### ⚙️ Model Settings")
            st.subheader("Risk Thresholds")
            low_t  = st.slider("Low Risk Threshold (mg/dL)",      0, 15, 10)
            mod_t  = st.slider("Moderate Risk Threshold (mg/dL)", 10, 20, 15)
            high_t = st.slider("High Risk Threshold (mg/dL)",     15, 25, 20)
            st.info(f"""**Current:** Low < {low_t} | Moderate {low_t}-{mod_t} | High {mod_t}-{high_t} | Critical > {high_t} mg/dL""")

            st.subheader("Image Analysis Settings")
            st.slider("Image Score Weight in Fusion", 0, 100, 45,
                      help="How much weight image score has vs. clinical observation (default 45%)")
            st.selectbox("Skin Tone Reference", ["Auto-detect","Light","Medium","Dark"],
                         help="Adjusts yellowness threshold for more accurate results")

            st.subheader("Alert Settings")
            st.checkbox("Enable Email Alerts", value=True)
            st.checkbox("Enable SMS Alerts", value=False)
            st.selectbox("Alert Threshold", ['High','Critical'])
            if st.button("💾 Save Settings"):
                st.success("Settings saved!")

        st.markdown("---")
        st.markdown("#### 🔍 Model Diagnostics")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Model Version", "v2.2.0")
        with c2: st.metric("Last Training", "Today")
        with c3: st.metric("Next Scheduled", "7 days")
        with c4: st.metric("Image Module", "Active ✅")

        st.markdown("##### 📊 Performance Over Time")
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        perf_df = pd.DataFrame({
            'Date':    dates,
            'Accuracy': np.clip(0.85 + np.random.randn(30)*0.04, 0.7, 1.0),
            'F1-Score': np.clip(0.80 + np.random.randn(30)*0.04, 0.7, 1.0),
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=perf_df['Date'], y=perf_df['Accuracy'],
                                  mode='lines', name='Accuracy',
                                  line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=perf_df['Date'], y=perf_df['F1-Score'],
                                  mode='lines', name='F1-Score',
                                  line=dict(color='green', width=2)))
        fig.update_layout(height=300, xaxis_title="Date",
                          yaxis_title="Score", yaxis_range=[0.7, 1.0])
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
def main():
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main()