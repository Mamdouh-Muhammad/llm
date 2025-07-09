import json
import io
import pandas as pd
import streamlit as st
from collections import Counter

st.set_page_config(page_title="Flow-to-JSONL Converter", layout="wide")
st.title("CSV → LLM fine-tune JSONL")

# ─────────────────────────────────────────────────────────────
# Helpers
def deduplicate_columns(columns):
    counts = Counter()
    new_cols = []
    for col in columns:
        col_clean = col.strip().replace("\xa0", " ")  # remove non-breaking space
        if counts[col_clean] == 0:
            new_cols.append(col_clean)
        else:
            new_cols.append(f"{col_clean}_{counts[col_clean]}")
        counts[col_clean] += 1
    return new_cols

def row_to_prompt(row, cols, sep="###"):
    return ", ".join(f"{col} is {row[col]}" for col in cols)


def convert_dataframe(df_in, cols, label_column, sep="###", end_token="<|endoftext|>"):
    records = []
    for _, row in df_in.iterrows():
        prompt = row_to_prompt(row, cols, sep)
        completion = str(row[label_column])
        records.append({
            "prompt": prompt + f" {sep} ",
            "completion": " " + completion + f" {end_token}"
        })
    return records

# ─────────────────────────────────────────────────────────────
# Upload CSV
uploaded = st.file_uploader("Upload CICFlowMeter CSV", type=["csv"])
if uploaded is None:
    st.stop()

df = pd.read_csv(uploaded)

# Clean and deduplicate column names
df.columns = deduplicate_columns(df.columns)

# ─────────────────────────────────────────────────────────────
# Column selection
st.subheader("Select feature columns for the prompt")
feature_cols = st.multiselect(
    "Columns",
    options=list(df.columns),
    default=[c for c in df.columns if c.lower() != "label"],
)

label_col = st.selectbox("Label column (completion)", options=df.columns)

# ─────────────────────────────────────────────────────────────
# Row-level editable preview
st.subheader("Edit rows if needed")
try:
    edited_df = st.data_editor(
        df[feature_cols + [label_col]].copy(),
        num_rows="dynamic",
        use_container_width=True,
        key="editor",
    )
except st.errors.StreamlitAPIException as e:
    st.error(f"Error in data editor: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Generate JSONL
if st.button("Generate JSONL"):
    records = convert_dataframe(edited_df, feature_cols, label_col)
    buf = io.StringIO("\n".join(json.dumps(r) for r in records))
    st.download_button(
        label="Download .jsonl",
        data=buf.getvalue(),
        file_name="flows.jsonl",
        mime="application/jsonl",
    )
    st.success(f"{len(records)} rows converted")
