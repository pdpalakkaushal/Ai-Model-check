"""
Streamlit App: AI Annotation & Facing Checker
------------------------------------------------
What it does:
1. Upload an Excel/CSV sheet with columns for Store, Category (optional),
   and an Image link. The image link can be a direct image URL, or a
   viewer-wrapped URL such as:
     https://view.shelfwatch.io/?url=https://storage.googleapis.com/.../photo.jpg
   The tool automatically extracts the real image URL from wrapper links.
2. Upload your CGC (Category / Group / Class) reference file - one row per
   SKU/class with a packshot image link. This is the "ground truth" the AI
   compares against. Reference images are grouped and cached by Category,
   so each unique category is only downloaded once even across many rows.
3. Claude Vision compares each shelf photo (which already has AI-generated
   annotations drawn on it) against the relevant reference packshots and:
   - judges whether the existing annotations are correct
   - gives an overall accuracy score
   - independently counts facings for every SKU it can confirm on the shelf
4. Results, plus a per-SKU facing breakdown and summary, are shown in the
   app and downloadable as a multi-sheet Excel report.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import base64
import io
import json
import re
import time
from urllib.parse import unquote

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
st.title("AI Annotation & Facing Checker (Claude Vision)")
st.caption(
    "Compare annotated shelf images against your CGC reference packshots to check "
    "annotation accuracy and count facings per SKU."
)

MODEL_OPTIONS = ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5-20251001"]
EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MAX_REFERENCE_IMAGES = 100  # hard cap so a single API call never fails on image count


# ----------------------------------------------------------------------------
# Helpers - URLs and images
# ----------------------------------------------------------------------------
def extract_actual_image_url(link: str) -> str:
    """
    Handles viewer-wrapped links such as:
      https://view.shelfwatch.io/?url=https://storage.googleapis.com/.../image.jpg
    and returns the direct underlying image URL. Links that don't match this
    pattern are returned unchanged.
    """
    if not isinstance(link, str) or not link.strip():
        return link
    match = re.search(r"[?&]url=(.+)$", link.strip())
    if match:
        candidate = unquote(match.group(1))
        if candidate.startswith("http"):
            return candidate
    return link.strip()


def fetch_image_bytes(url: str, timeout: int = 25, retries: int = 2):
    """
    Downloads an image and returns (bytes, media_type) or (None, error_message).
    Tries the direct/extracted image URL first. If that fails and the original
    link looked like a wrapper URL, falls back to requesting the original link
    directly as a second attempt.
    """
    if not isinstance(url, str) or not url.strip():
        return None, "Empty or invalid link"

    direct_url = extract_actual_image_url(url)
    candidates = [direct_url]
    if direct_url != url.strip():
        candidates.append(url.strip())

    last_error = "Unknown error"
    for candidate_url in candidates:
        for attempt in range(retries):
            try:
                resp = requests.get(candidate_url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                img_bytes = resp.content

                if "text/html" in content_type.lower():
                    last_error = "This URL returned a web page, not an image file directly."
                    break

                try:
                    img = Image.open(io.BytesIO(img_bytes))
                    fmt = (img.format or "JPEG").upper()
                    media_type = {
                        "JPEG": "image/jpeg", "JPG": "image/jpeg", "PNG": "image/png",
                        "WEBP": "image/webp", "GIF": "image/gif",
                    }.get(fmt, "image/jpeg")
                    buf = io.BytesIO()
                    if img.mode in ("RGBA", "P") and media_type == "image/jpeg":
                        img = img.convert("RGB")
                    img.save(buf, format="JPEG" if media_type == "image/jpeg" else fmt)
                    img_bytes = buf.getvalue()
                except Exception:
                    if "png" in content_type:
                        media_type = "image/png"
                    elif "webp" in content_type:
                        media_type = "image/webp"
                    else:
                        media_type = "image/jpeg"

                return img_bytes, media_type
            except Exception as e:
                last_error = str(e)
                time.sleep(0.5)
                continue
    return None, last_error


def to_b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")


def extract_json(text: str):
    """Pulls the first JSON object out of a model response, stripping ```json fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def to_excel_bytes(sheets: dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = str(name)[:31] if name else "Sheet1"
            if df is None or df.empty:
                df = pd.DataFrame([{"Info": "No rows"}])
            df.to_excel(writer, sheet_name=safe_name, index=False)
    return output.getvalue()


def read_any(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


# ----------------------------------------------------------------------------
# Helpers - prompting and the Claude call
# ----------------------------------------------------------------------------
def build_prompt(cgc_labels, extra_instruction):
    if cgc_labels:
        ref_line = (
            f"You were given {len(cgc_labels)} reference packshot image(s), each labelled with its "
            f"SKU/class name: {', '.join(cgc_labels)}. These are the ground-truth appearance of these SKUs.\n"
        )
    else:
        ref_line = "No reference packshot images were provided; use general retail-shelf knowledge.\n"

    extra_line = f"Additional instruction: {extra_instruction.strip()}\n" if extra_instruction and extra_instruction.strip() else ""

    return f"""You are a retail shelf-audit QA expert. You are shown:
1. Reference packshot image(s) of known SKUs (ground truth appearance), each labelled by name.
2. A shelf photo that already has AI-generated annotations drawn on it (bounding boxes and/or
   labels marking SKUs on the shelf).

{ref_line}{extra_line}
Tasks:
A. Judge whether the existing annotations on the shelf photo correctly identify the SKUs shown,
   using the reference packshots as ground truth.
B. Independently count, for every reference SKU you can visually confirm on the shelf, how many
   facings (individual visible units) of that SKU appear in the photo. Only use SKU names from
   the reference label list above - if a facing does not clearly match any reference SKU, skip it
   rather than inventing a name.

Respond with ONLY a single JSON object (no markdown, no extra text) in exactly this shape:

{{
  "overall_verdict": "correct" | "partially_correct" | "incorrect",
  "accuracy_score": <integer 0-100, how correct the existing annotations look>,
  "correct_annotations": ["short description", ...],
  "incorrect_annotations": ["short description of what is wrong", ...],
  "sku_facings": [
    {{"sku_name": "<must match one of the reference labels exactly>", "facing_count": <integer>}}
  ],
  "remarks": "1-2 sentence summary in simple language"
}}

Only output the JSON object, nothing else."""


def call_claude(client, model, cgc_images, cgc_labels, shelf_b64, shelf_media_type, extra_instruction, max_tokens=1500, retries=2):
    content = []
    for label, (b64, media_type) in zip(cgc_labels, cgc_images):
        content.append({"type": "text", "text": f"Reference packshot - {label}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}})
    content.append({"type": "text", "text": "Shelf photo to audit (already annotated):"})
    content.append({"type": "image", "source": {"type": "base64", "media_type": shelf_media_type, "data": shelf_b64}})
    content.append({"type": "text", "text": build_prompt(cgc_labels, extra_instruction)})

    last_err = None
    for attempt in range(retries):
        try:
            resp = client.messages.create(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": content}])
            text = "".join(block.text for block in resp.content if block.type == "text")
            return extract_json(text)
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise last_err


# ----------------------------------------------------------------------------
# Sidebar - settings
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Anthropic API Key", type="password")
    model = st.selectbox(
        "Model", MODEL_OPTIONS, index=0,
        help="Sonnet is a strong default balance of speed, cost, and accuracy. Use Opus for the highest accuracy.",
    )

    st.divider()
    st.subheader("CGC Reference Images")
    st.caption("Category / Group / Class reference packshots the AI compares against.")

    cgc_source = st.radio(
        "How will you provide reference images?",
        ["Upload CGC sheet (Category/Class + packshot link)", "Upload individual image files"],
    )

    max_ref_per_category = st.slider(
        "Max reference images per category", 5, MAX_REFERENCE_IMAGES, 30, step=5,
        help="Keeps each API call fast and within limits. Increase only if a category has many SKUs you need checked.",
    )

    cgc_by_category = {}
    cgc_flat = []

    if cgc_source.startswith("Upload CGC sheet"):
        cgc_table_file = st.file_uploader("CGC sheet", type=["xlsx", "xls", "csv"], key="cgc_table")
        if cgc_table_file is not None:
            cgc_df = read_any(cgc_table_file)
            st.caption(f"{len(cgc_df)} rows loaded")

            cgc_cols = list(cgc_df.columns)
            label_guess = next(
                (c for c in cgc_cols if any(k in c.lower() for k in ["class_name", "class name", "sku", "label"])),
                cgc_cols[0],
            )
            url_guess = next(
                (c for c in cgc_cols if any(k in c.lower() for k in ["class_image", "packshot", "image", "link", "url", "view"])),
                cgc_cols[-1],
            )
            category_guess = next((c for c in cgc_cols if "categ" in c.lower()), None)

            cgc_label_col = st.selectbox("SKU/Class name column", cgc_cols, index=cgc_cols.index(label_guess), key="cgc_label_col")
            cgc_url_col = st.selectbox("Packshot image link column", cgc_cols, index=cgc_cols.index(url_guess), key="cgc_url_col")
            cat_options = ["(none)"] + cgc_cols
            cat_default_idx = cat_options.index(category_guess) if category_guess in cat_options else 0
            category_col_cgc = st.selectbox("Category column (for filtering, optional)", cat_options, index=cat_default_idx, key="cgc_cat_col")
            if category_col_cgc == "(none)":
                category_col_cgc = None

            if st.button("Load CGC reference images"):
                with st.spinner("Downloading CGC reference images... this may take a while for large sheets."):
                    work_df = cgc_df.drop_duplicates(subset=[cgc_label_col])
                    loaded_count, failed_count = 0, 0
                    new_by_category, new_flat = {}, []

                    if category_col_cgc:
                        for cat_value, group in work_df.groupby(category_col_cgc):
                            bucket = []
                            for _, r in group.head(max_ref_per_category).iterrows():
                                b, media_or_err = fetch_image_bytes(str(r[cgc_url_col]))
                                if b is None:
                                    failed_count += 1
                                    continue
                                bucket.append((str(r[cgc_label_col]), (to_b64(b), media_or_err)))
                                loaded_count += 1
                            new_by_category[str(cat_value)] = bucket
                    else:
                        for _, r in work_df.head(max_ref_per_category).iterrows():
                            b, media_or_err = fetch_image_bytes(str(r[cgc_url_col]))
                            if b is None:
                                failed_count += 1
                                continue
                            new_flat.append((str(r[cgc_label_col]), (to_b64(b), media_or_err)))
                            loaded_count += 1

                    st.session_state["cgc_by_category"] = new_by_category
                    st.session_state["cgc_flat"] = new_flat
                    msg = f"Loaded {loaded_count} reference image(s)"
                    if failed_count:
                        msg += f" - {failed_count} failed to download"
                    st.success(msg)

        cgc_by_category = st.session_state.get("cgc_by_category", {})
        cgc_flat = st.session_state.get("cgc_flat", [])
        if cgc_by_category:
            for cat, bucket in cgc_by_category.items():
                st.caption(f"'{cat}': {len(bucket)} reference image(s)")
        elif cgc_flat:
            st.caption(f"{len(cgc_flat)} reference image(s) ready")

    else:
        cgc_files = st.file_uploader("Reference images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
        if cgc_files:
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
                cgc_flat.append((label, (to_b64(img_bytes), media_type)))

    st.divider()
    with st.expander("🔧 Test a single image link"):
        test_link = st.text_input("Paste an image link to test", key="test_link")
        if st.button("Fetch & preview"):
            if test_link:
                img_bytes, media_type_or_err = fetch_image_bytes(test_link)
                if img_bytes:
                    st.image(img_bytes, caption=f"Fetched OK ({media_type_or_err})", use_container_width=True)
                else:
                    st.error(f"Could not fetch this link: {media_type_or_err}")


# ----------------------------------------------------------------------------
# Main - shelf image sheet upload and processing
# ----------------------------------------------------------------------------
st.subheader("1. Upload Shelf Image Sheet")
excel_file = st.file_uploader(
    "Excel/CSV file with Store, Category (optional), and Image link columns",
    type=["xlsx", "xls", "csv"],
)

if excel_file:
    df = read_any(excel_file)
    st.caption("Preview:")
    st.dataframe(df.head(10), use_container_width=True)

    cols = list(df.columns)
    store_guess = next((c for c in cols if "store" in c.lower()), cols[0])
    category_guess_shelf = next((c for c in cols if "categ" in c.lower()), None)
    url_guess_shelf = next((c for c in cols if any(k in c.lower() for k in ["image", "link", "view", "url"])), cols[-1])
    date_guess = next((c for c in cols if "date" in c.lower()), None)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        store_col = st.selectbox("Store column", cols, index=cols.index(store_guess))
    with c2:
        cat_options_shelf = ["(none)"] + cols
        cat_default = cat_options_shelf.index(category_guess_shelf) if category_guess_shelf in cat_options_shelf else 0
        category_col_shelf = st.selectbox("Category column (optional)", cat_options_shelf, index=cat_default)
        if category_col_shelf == "(none)":
            category_col_shelf = None
    with c3:
        url_col = st.selectbox("Image link column", cols, index=cols.index(url_guess_shelf))
    with c4:
        date_options = ["(none)"] + cols
        date_default = date_options.index(date_guess) if date_guess in date_options else 0
        date_col = st.selectbox("Date column (optional)", date_options, index=date_default)
        if date_col == "(none)":
            date_col = None

    st.subheader("2. Process")
    max_rows = st.number_input(
        "Rows to process (limit while testing to control API cost)",
        min_value=1, max_value=int(len(df)), value=min(10, int(len(df))),
    )
    extra_instruction = st.text_input("Extra instruction for the model (optional)", placeholder="e.g. focus only on the top shelf")

    if not cgc_by_category and not cgc_flat:
        st.warning("No CGC reference images loaded yet - accuracy checking works best with reference packshots. You can still run without them.")

    run = st.button("🚀 Run AI Check", type="primary", disabled=not api_key)
    if not api_key:
        st.info("Enter your Anthropic API key in the sidebar first.")

    if run:
        if anthropic is None:
            st.error("The `anthropic` package is not installed. Run: pip install anthropic")
            st.stop()

        client = anthropic.Anthropic(api_key=api_key)
        results = []
        sku_facing_rows = []
        progress = st.progress(0)
        status = st.empty()
        subset = df.head(int(max_rows)).reset_index(drop=True)

        for i, row in subset.iterrows():
            store_val = row[store_col]
            cat_val = row[category_col_shelf] if category_col_shelf else ""
            status.write(f"Processing row {i + 1}/{len(subset)} — Store: {store_val}")

            if category_col_shelf and cgc_by_category:
                ref_bucket = cgc_by_category.get(str(cat_val), [])
                if not ref_bucket and cgc_flat:
                    ref_bucket = cgc_flat
            elif cgc_flat:
                ref_bucket = cgc_flat
            elif cgc_by_category:
                ref_bucket = next(iter(cgc_by_category.values()), [])
            else:
                ref_bucket = []

            ref_bucket = ref_bucket[:MAX_REFERENCE_IMAGES]
            cgc_labels = [l for l, _ in ref_bucket]
            cgc_images_only = [img for _, img in ref_bucket]

            base_result = {
                "Store": store_val,
                "Category": cat_val,
                "Date": row[date_col] if date_col else "",
                "Image Link": row[url_col],
            }

            img_bytes, media_type_or_err = fetch_image_bytes(str(row[url_col]))
            if img_bytes is None:
                base_result.update({
                    "overall_verdict": "error",
                    "accuracy_score": None,
                    "correct_annotations": "",
                    "incorrect_annotations": f"Image fetch failed: {media_type_or_err}",
                    "total_facings": None,
                    "remarks": "Could not download the image for this row.",
                })
                results.append(base_result)
                progress.progress((i + 1) / len(subset))
                continue

            shelf_b64 = to_b64(img_bytes)
            try:
                ai_json = call_claude(client, model, cgc_images_only, cgc_labels, shelf_b64, media_type_or_err, extra_instruction)
                sku_facings = ai_json.get("sku_facings", []) or []
                base_result.update({
                    "overall_verdict": ai_json.get("overall_verdict"),
                    "accuracy_score": ai_json.get("accuracy_score"),
                    "correct_annotations": "; ".join(ai_json.get("correct_annotations", []) or []),
                    "incorrect_annotations": "; ".join(ai_json.get("incorrect_annotations", []) or []),
                    "total_facings": sum(int(s.get("facing_count", 0) or 0) for s in sku_facings),
                    "remarks": ai_json.get("remarks"),
                })
                for s in sku_facings:
                    sku_facing_rows.append({
                        "Store": store_val,
                        "Category": cat_val,
                        "SKU": s.get("sku_name"),
                        "Facing Count": s.get("facing_count"),
                    })
            except Exception as e:
                base_result.update({
                    "overall_verdict": "error",
                    "accuracy_score": None,
                    "correct_annotations": "",
                    "incorrect_annotations": f"AI call failed: {e}",
                    "total_facings": None,
                    "remarks": "The model call failed for this row.",
                })

            results.append(base_result)
            progress.progress((i + 1) / len(subset))
            time.sleep(0.2)

        status.write("✅ Done!")
        result_df = pd.DataFrame(results)
        facing_df = pd.DataFrame(sku_facing_rows)

        st.subheader("3. Results")
        st.dataframe(result_df, use_container_width=True)

        valid_scores = result_df["accuracy_score"].dropna()
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Accuracy Score", f"{valid_scores.mean():.1f}" if len(valid_scores) else "N/A")
        m2.metric("Images Processed", len(result_df))
        m3.metric("Errors", int((result_df["overall_verdict"] == "error").sum()))

        facing_summary = pd.DataFrame()
        if not facing_df.empty:
            st.subheader("SKU Facing Summary (across all processed images)")
            facing_summary = (
                facing_df.groupby("SKU", as_index=False)["Facing Count"]
                .sum()
                .sort_values("Facing Count", ascending=False)
            )
            st.dataframe(facing_summary, use_container_width=True, hide_index=True)

            st.subheader("SKU Facing Details (per image)")
            st.dataframe(facing_df, use_container_width=True, hide_index=True)

        out_bytes = to_excel_bytes({
            "Results": result_df,
            "SKU Facing Details": facing_df,
            "SKU Facing Summary": facing_summary,
        })
        st.download_button(
            "⬇️ Download Results (Excel)",
            data=out_bytes,
            file_name="annotation_check_results.xlsx",
            mime=EXCEL_MIME,
        )
else:
    st.info("Upload a shelf image sheet above to get started.")
