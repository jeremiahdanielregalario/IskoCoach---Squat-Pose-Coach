
# evaluation/annotate.py
import streamlit as st
import json
import os
from pathlib import Path

st.title("IskoCoach — Annotate Ground Truth")

log_dir = "session_logs"
log_files = list(Path(log_dir).glob("session_*.json"))
sel = st.selectbox("Select session log", [""] + [str(p) for p in log_files])

if sel:
    with open(sel, "r") as f:
        session = json.load(f)
    st.write(f"User: {session.get('user')}  | Reps: {session.get('reps')}")
    frames = session.get("frames", [])
    st.dataframe(frames)

    if "gt_marks" not in st.session_state:
        st.session_state.gt_marks = []

    if st.button("Mark current last frame as rep"):
        idx = len(frames)-1
        st.session_state.gt_marks.append({"frame_index": idx, "timestamp": frames[idx]["timestamp"]})

    if st.button("Export ground truth"):
        out = {
            "session_file": sel,
            "gt": st.session_state.gt_marks
        }
        os.makedirs("evaluation/data", exist_ok=True)
        outpath = f"evaluation/data/gt_{Path(sel).stem}.json"
        with open(outpath, "w") as fw:
            json.dump(out, fw, indent=2)
        st.success(f"Saved to {outpath}")
        st.session_state.gt_marks = []
