import streamlit as st
import pandas as pd
import numpy as np
import easyocr
import cv2
from PIL import Image
import io

st.set_page_config(page_title="Picklist OCR App", layout="wide")
st.title("📦 Picklist OCR App")
st.caption("Upload or capture a picklist photo. Extract Item Numbers, Qty Ordered, and Qty Picked.")

# Load EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Function to process the image
def process_image(img):
    img_np = np.array(img.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # OCR
    result = reader.readtext(gray, detail=1)

    # Extract text
    extracted_text = [res[1] for res in result]
    
    # Build DataFrame logic
    data = []
    for i, text in enumerate(extracted_text):
        if text.isdigit() and len(text) == 6:  # likely Item Number
            try:
                qty_ordered = int(extracted_text[i+4])  # Sum of Quantity
                actual_collected = extracted_text[i+5]
                # Analyze actual_collected
                if '✓' in actual_collected:
                    qty_picked = qty_ordered
                elif '✗' in actual_collected:
                    qty_picked = 0
                else:
                    try:
                        qty_picked = int(''.join(filter(str.isdigit, actual_collected)))
                    except:
                        qty_picked = 0
                data.append({
                    "Item Number": text,
                    "Qty Ordered": qty_ordered,
                    "Qty Picked": qty_picked
                })
            except:
                continue
    return pd.DataFrame(data)

# Upload or capture
option = st.radio("📸 Capture or 📂 Upload Pick List", ("Capture picklist using mobile camera", "Upload picklist photo"))

if option == "Capture picklist using mobile camera":
    img_file = st.camera_input("Take a photo of the picklist")
else:
    img_file = st.file_uploader("Or upload picklist photo", type=["jpg", "jpeg", "png"])

if img_file:
    st.image(img_file, caption="Uploaded Picklist", use_container_width=True)
    img = Image.open(img_file)

    df = process_image(img)

    if df.empty:
        st.error("Could not detect any rows. Try uploading a clearer image.")
    else:
        st.success("✅ Data extracted successfully!")
        st.dataframe(df, use_container_width=True)

        # Download Excel
        excel_io = io.BytesIO()
        df.to_excel(excel_io, index=False)
        st.download_button(
            "📥 Download Excel",
            data=excel_io.getvalue(),
            file_name="picklist.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
