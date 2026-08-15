"""
Streamlit App: AI Annotation & Facing Checker
------------------------------------------------
Kya karta hai:
1. Excel file upload karo jisme columns hon: Date, Store ID, aur Image/View link
   (ye image link already annotated hoti hai - shelf photo par bounding box/labels).
2. CGC (Category Group Class) reference images upload karo - ye "sahi" SKU ka
   reference hai jisse AI model comparison karega.
3. Claude Vision (Anthropic API) har row ki image ko CGC reference se compare karke:
   - batata hai annotation sahi hai ya galat
   - overall accuracy score deta hai
   - self-brand SKU ki facing count aur competitor SKU ki facing count deta hai
4. Result table dikhta hai aur Excel me download bhi ho sakta hai.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import base64
import io
import json
import re
import time

import pandas as pd
import requests
import streamlit as st
from PIL import Image

try:
    import anthropic
except ImportError:
    anthropic = None


# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Annotation & Facing Checker", layout="wide")
st.title("🧠 AI Annotation & Facing Checker (Claude Vision)")
st.caption(
    "Excel me diye gaye annotated shelf images ko CGC reference images se compare "
    "karke annotation correctness aur SKU facing count nikalta hai."
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def fetch_image_bytes(url: str, timeout: int = 20):
    """Download image bytes from a URL. Returns (bytes, media_type) or (None, error_msg)."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        img_bytes = resp.content

        # Validate + normalize via PIL, and figure out a safe media_type
        try:
            img = Image.open(io.BytesIO(img_bytes))
            fmt = (img.format or "JPEG").upper()
            media_type = {
                "JPEG": "image/jpeg",
                "JPG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
                "GIF": "image/gif",
            }.get(fmt, "image/jpeg")
            # Re-encode to be safe (some CDNs serve odd containers)
            buf = io.BytesIO()
            if img.mode in ("RGBA", "P") and media_type == "image/jpeg":
                img = img.convert("RGB")
            img.save(buf, format="JPEG" if media_type == "image/jpeg" else fmt)
            img_bytes = buf.getvalue()
        except Exception:
            # fallback to content-type header guess if PIL fails
            if "png" in content_type:
                media_type = "image/png"
            elif "webp" in content_type:
                media_type = "image/webp"
            else:
                media_type = "image/jpeg"

        return img_bytes, media_type
    except Exception as e:
        return None, str(e)


def to_b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")


def extract_json(text: str):
    """Pull the first JSON object out of a model response, stripping ```json fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def build_prompt(sku_focus: str, cgc_labels: list) -> str:
    focus_line = (
        f'Especially count and report facings for the SKU "{sku_focus}" separately '
        f"under self_facing_count / competitor_facing_count.\n"
        if sku_focus
        else ""
    )
    ref_line = (
        f"Reference (CGC - Category Group Class) images provided are labelled: "
        f"{', '.join(cgc_labels)}. These represent the CORRECT ground-truth appearance "
        f"of the SKUs/products.\n"
        if cgc_labels
        else "No CGC reference images were provided; judge annotation quality using general "
        "retail-shelf knowledge.\n"
    )
    return f"""You are a retail shelf-audit QA expert. You will be shown:
1. One or more CGC reference images (ground truth of correct SKU/category appearance).
2. A shelf photo that already has AI-generated annotations drawn on it (bounding boxes
   and/or labels marking SKUs, both the client's own ("self") brand and competitor brands).

{ref_line}{focus_line}
Carefully compare the annotations in the shelf photo against the CGC reference images and
general shelf-audit logic. Then respond with ONLY a single JSON object (no markdown, no
extra text) in exactly this shape:

{{
  "overall_verdict": "correct" | "partially_correct" | "incorrect",
  "accuracy_score": <integer 0-100, how correct the annotations look>,
  "correct_annotations": ["short description", ...],
  "incorrect_annotations": ["short description of what is wrong", ...],
  "self_facing_count": <integer, total facings of the self/own brand SKU visible>,
  "competitor_facing_count": <integer, total facings of competitor SKU(s) visible>,
  "remarks": "1-2 sentence summary in simple language"
}}

Only output the JSON object, nothing else."""


def call_claude(client, model, api_key_ok, cgc_images, cgc_labels, shelf_img_b64,
                 shelf_media_type, sku_focus):
    content = []
    for label, (b64, media_type) in zip(cgc_labels, cgc_images):
        content.append({"type": "text", "text": f"CGC reference image - {label}:"})
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64},
            }
        )
    content.append({"type": "text", "text": "Shelf photo to audit (already annotated):"})
    content.append(
        {
            "type": "image",
            "source": {"type": "base64", "media_type": shelf_media_type, "data": shelf_img_b64},
        }
    )
    content.append({"type": "text", "text": build_prompt(sku_focus, cgc_labels)})

    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return extract_json(text)


# ----------------------------------------------------------------------------
# Sidebar - configuration
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input("Anthropic API Key", type="password")
    model = st.selectbox(
        "Model",
        ["claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5-20251001"],
        index=0,
        help="Sonnet ek accha balance hai speed aur accuracy ka. Zyada accuracy ke liye Opus try karein.",
    )

    sku_focus = st.text_input(
        "Focus SKU naam (optional)",
        placeholder="e.g. XYZ",
        help="Agar bharenge to model specifically is SKU ki facing count par focus karega.",
    )

    st.divider()
    st.subheader("📌 CGC Reference Images")
    st.caption("Category Group Class - ye 'sahi' reference images hain jo tool me already upload hain.")
    cgc_mode = st.radio(
        "CGC images kaise denge?",
        ["Excel/CSV file (with image link)", "Direct image files upload karo"],
        help="Excel/CSV wale option me ek file dein jisme SKU/label naam aur uski image ka link ho — "
        "bilkul jaise main shelf-image excel me hota hai.",
    )

    cgc_data = []  # list of (label, (b64, media_type))

    if cgc_mode == "Excel/CSV file (with image link)":
        cgc_table_file = st.file_uploader(
            "CGC file upload karein (Excel/CSV — SKU/label naam + image link column)",
            type=["xlsx", "xls", "csv"],
            key="cgc_table",
        )
        if cgc_table_file is not None:
            if cgc_table_file.name.lower().endswith(".csv"):
                cgc_df = pd.read_csv(cgc_table_file)
            else:
                cgc_df = pd.read_excel(cgc_table_file)

            st.caption("Preview:")
            st.dataframe(cgc_df.head(10), use_container_width=True)

            cgc_cols = list(cgc_df.columns)
            label_guess = next(
                (c for c in cgc_cols if any(k in c.lower() for k in ["sku", "label", "name", "category", "cgc"])),
                cgc_cols[0],
            )
            url_guess_cgc = next(
                (c for c in cgc_cols if any(k in c.lower() for k in ["view", "image", "link", "url"])),
                cgc_cols[-1],
            )
            cgc_label_col = st.selectbox("SKU/Label column", cgc_cols, index=cgc_cols.index(label_guess), key="cgc_label_col")
            cgc_url_col = st.selectbox("Image link column", cgc_cols, index=cgc_cols.index(url_guess_cgc), key="cgc_url_col")

            cgc_max_rows = st.number_input(
                "Kitni CGC rows use karni hain",
                min_value=1,
                max_value=int(len(cgc_df)),
                value=min(20, int(len(cgc_df))),
                key="cgc_max_rows",
            )

            if st.button("📥 CGC images load karo", key="load_cgc_btn"):
                with st.spinner("CGC images download ho rahi hain..."):
                    loaded, failed = [], []
                    for _, crow in cgc_df.head(int(cgc_max_rows)).iterrows():
                        c_url = str(crow[cgc_url_col])
                        c_label = str(crow[cgc_label_col])
                        c_bytes, c_media_or_err = fetch_image_bytes(c_url)
                        if c_bytes is None:
                            failed.append((c_label, c_media_or_err))
                            continue
                        loaded.append((c_label, (to_b64(c_bytes), c_media_or_err)))
                    st.session_state["cgc_loaded_data"] = loaded
                    if failed:
                        st.warning(f"{len(failed)} CGC image(s) load nahi ho payi (link check karein).")
                    st.success(f"{len(loaded)} CGC reference image(s) load ho gayi.")

        cgc_data = st.session_state.get("cgc_loaded_data", [])
        if cgc_data:
            st.caption(f"✅ {len(cgc_data)} CGC image(s) ready: " + ", ".join(l for l, _ in cgc_data))

    else:
        cgc_files = st.file_uploader(
            "CGC images upload karein (multiple allowed)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )
        if cgc_files:
            st.caption("Har image ko ek naam/label dein (SKU/category naam):")
            for i, f in enumerate(cgc_files):
                default_label = f.name.rsplit(".", 1)[0]
                label = st.text_input(f"Label for '{f.name}'", value=default_label, key=f"cgc_label_{i}")
                img_bytes = f.read()
                try:
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    img_bytes = buf.getvalue()
                    media_type = "image/jpeg"
                except Exception:
                    media_type = f.type or "image/jpeg"
                cgc_data.append((label, (to_b64(img_bytes), media_type)))


# ----------------------------------------------------------------------------
# Main - excel upload
# ----------------------------------------------------------------------------
st.subheader("1️⃣ Excel File Upload Karein")
excel_file = st.file_uploader("Excel file (.xlsx) - Date, Store ID, Image/View link columns ke saath", type=["xlsx", "xls"])

if excel_file:
    df = pd.read_excel(excel_file)
    st.write("Preview:")
    st.dataframe(df.head(10), use_container_width=True)

    cols = list(df.columns)
    c1, c2, c3 = st.columns(3)
    with c1:
        date_col = st.selectbox("Date column", cols, index=cols.index("Date") if "Date" in cols else 0)
    with c2:
        store_col = st.selectbox(
            "Store ID column", cols, index=cols.index("Store ID") if "Store ID" in cols else 0
        )
    with c3:
        url_guess = next((c for c in cols if "view" in c.lower() or "image" in c.lower() or "link" in c.lower()), cols[0])
        url_col = st.selectbox("Image/View link column", cols, index=cols.index(url_guess))

    st.subheader("2️⃣ Process")
    max_rows = st.number_input(
        "Kitni rows process karni hain (testing ke liye limit rakh sakte hain)",
        min_value=1,
        max_value=int(len(df)),
        value=min(10, int(len(df))),
    )

    run = st.button("🚀 AI se Check Karo", type="primary", disabled=not api_key)
    if not api_key:
        st.info("Pehle sidebar me Anthropic API key daalein.")

    if run:
        if anthropic is None:
            st.error("`anthropic` package installed nahi hai. `pip install anthropic` chalayein.")
            st.stop()

        client = anthropic.Anthropic(api_key=api_key)
        cgc_labels = [lbl for lbl, _ in cgc_data]
        cgc_images_only = [img for _, img in cgc_data]

        results = []
        progress = st.progress(0)
        status = st.empty()
        subset = df.head(int(max_rows)).reset_index(drop=True)

        for i, row in subset.iterrows():
            status.write(f"Processing row {i + 1}/{len(subset)} — Store: {row[store_col]}")
            url = str(row[url_col])
            img_bytes, media_type_or_err = fetch_image_bytes(url)

            base_result = {
                "Date": row[date_col],
                "Store ID": row[store_col],
                "Image URL": url,
            }

            if img_bytes is None:
                base_result.update(
                    {
                        "overall_verdict": "error",
                        "accuracy_score": None,
                        "correct_annotations": "",
                        "incorrect_annotations": f"Image fetch failed: {media_type_or_err}",
                        "self_facing_count": None,
                        "competitor_facing_count": None,
                        "remarks": "Image download nahi ho payi.",
                    }
                )
                results.append(base_result)
                progress.progress((i + 1) / len(subset))
                continue

            shelf_b64 = to_b64(img_bytes)
            try:
                ai_json = call_claude(
                    client,
                    model,
                    bool(api_key),
                    cgc_images_only,
                    cgc_labels,
                    shelf_b64,
                    media_type_or_err,
                    sku_focus,
                )
                base_result.update(
                    {
                        "overall_verdict": ai_json.get("overall_verdict"),
                        "accuracy_score": ai_json.get("accuracy_score"),
                        "correct_annotations": "; ".join(ai_json.get("correct_annotations", []) or []),
                        "incorrect_annotations": "; ".join(ai_json.get("incorrect_annotations", []) or []),
                        "self_facing_count": ai_json.get("self_facing_count"),
                        "competitor_facing_count": ai_json.get("competitor_facing_count"),
                        "remarks": ai_json.get("remarks"),
                    }
                )
            except Exception as e:
                base_result.update(
                    {
                        "overall_verdict": "error",
                        "accuracy_score": None,
                        "correct_annotations": "",
                        "incorrect_annotations": f"AI call failed: {e}",
                        "self_facing_count": None,
                        "competitor_facing_count": None,
                        "remarks": "AI model se response process nahi hua.",
                    }
                )

            results.append(base_result)
            progress.progress((i + 1) / len(subset))
            time.sleep(0.2)  # gentle pacing

        status.write("✅ Done!")
        result_df = pd.DataFrame(results)

        st.subheader("3️⃣ Results")
        st.dataframe(result_df, use_container_width=True)

        valid_scores = result_df["accuracy_score"].dropna()
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Accuracy Score", f"{valid_scores.mean():.1f}" if len(valid_scores) else "N/A")
        m2.metric("Total Self Facings", int(result_df["self_facing_count"].fillna(0).sum()))
        m3.metric("Total Competitor Facings", int(result_df["competitor_facing_count"].fillna(0).sum()))

        out_buf = io.BytesIO()
        result_df.to_excel(out_buf, index=False, engine="openpyxl")
        out_buf.seek(0)
        st.download_button(
            "⬇️ Download Results (Excel)",
            data=out_buf,
            file_name="annotation_check_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
else:
    st.info("Shuru karne ke liye Excel file upload karein.")
