import streamlit as st
import pandas as pd
import numpy as np
import easyocr
import cv2
from PIL import Image
import io

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)

st.set_page_config(page_title="Picklist OCR App", layout="wide")
st.title("📦 Picklist OCR App")
st.caption("Capture or upload a picklist photo. Detect handwritten ✓/✗ and quantities, then download Excel.")

# Upload or capture image
option = st.radio("📸 Capture or 📂 Upload Pick List", ("Capture picklist using mobile camera", "Upload picklist photo"))

if option == "Capture picklist using mobile camera":
    img_file = st.camera_input("Take a photo of the picklist")
else:
    img_file = st.file_uploader("Or upload picklist photo", type=["jpg", "jpeg", "png"])

if img_file is not None:
    st.image(img_file, caption="Selected Picklist", use_container_width=True)

    image = Image.open(img_file).convert("RGB")
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Preprocess to detect table rows
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 50 and w > 100:  # Adjust size thresholds for rows
            row_img = img_np[y:y+h, :]
            rows.append((y, row_img))

    rows.sort(key=lambda x: x[0])  # Sort rows from top to bottom

    data = []
    for idx, (y_pos, row_img) in enumerate(rows):
        row_gray = cv2.cvtColor(row_img, cv2.COLOR_RGB2GRAY)
        ocr_results = reader.readtext(row_gray, detail=0)
        clean_text = [t.strip() for t in ocr_results if t.strip()]

        if len(clean_text) < 3:
            continue  # skip incomplete rows

        try:
            item_code = next(t for t in clean_text if t.isdigit() and len(t) >= 5)
            qty_ordered = next(t for t in clean_text if t.isdigit() and len(t) <= 4)
            qty_picked = 0

            # Check for ✓, ✗, or handwritten qty
            for t in clean_text:
                lower_t = t.lower()
                if lower_t in ["✓", "✔", "tick"]:
                    qty_picked = qty_ordered
                    break
                elif lower_t in ["✗", "x", "cross"]:
                    qty_picked = 0
                    break
                elif lower_t.isdigit() and len(lower_t) <= 4:
                    qty_picked = lower_t
                    break

            data.append({
                "Item Code": item_code,
                "Qty Ordered": qty_ordered,
                "Qty Picked": qty_picked
            })
        except StopIteration:
            continue

    if data:
        df = pd.DataFrame(data)
        st.success("✅ Text extracted successfully!")
        st.dataframe(df, use_container_width=True)

        # Download Excel
        excel_bytes = io.BytesIO()
        df.to_excel(excel_bytes, index=False)
        st.download_button(
            "📥 Download Excel",
            data=excel_bytes.getvalue(),
            file_name="picklist.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.error("⚠️ Could not detect rows properly. Try uploading a clearer photo.")
