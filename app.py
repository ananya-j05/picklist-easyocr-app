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
    # Convert to OpenCV format
    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    # Preprocess image for better OCR
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # OCR with EasyOCR
    results = reader.readtext(thresh, detail=0)

    if not results:
        st.error("No text detected. Please try a clearer image.")
    else:
        st.success("Text detected successfully!")
        detected_text = "\n".join(results)
        st.text_area("Detected Text", detected_text, height=300)

        # Parse text to find item codes and quantities
        data = []
        lines = detected_text.split("\n")
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                item_code = parts[0]
                qty_ordered = parts[1]
                
                qty_picked = 0  # Default: nothing picked

                # Look for tick or cross or handwritten quantity
                for part in parts[2:]:
                    cleaned = part.strip().lower()
                    if cleaned in ["✓", "✔", "tick"]:  # tick detected
                        qty_picked = qty_ordered
                        break
                    elif cleaned in ["✗", "x", "cross"]:  # cross detected
                        qty_picked = 0
                        break
                    elif cleaned.isdigit():  # handwritten quantity
                        qty_picked = cleaned
                        break

                data.append({
                    "Item Code": item_code,
                    "Qty Ordered": qty_ordered,
                    "Qty Picked": qty_picked
                })

        df = pd.DataFrame(data)
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
