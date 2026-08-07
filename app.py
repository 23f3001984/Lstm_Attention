import utils.transport
import utils.session
import streamlit as st

from predict import MCQSolver

st.set_page_config(page_title="Smart MCQ Solver — BiLSTM+Attention", page_icon="🧠")

st.title("🧠 Smart MCQ Solver")
st.caption("BiLSTM + Attention model trained from scratch (MAP@3 competition model)")


@st.cache_resource
def load_solver():
    return MCQSolver()


with st.spinner("Loading model..."):
    solver = load_solver()

st.subheader("Question")
prompt = st.text_area("Prompt", placeholder="Type the MCQ question here...", height=100)

st.subheader("Options")
col1, col2 = st.columns(2)
with col1:
    opt_a = st.text_input("A")
    opt_b = st.text_input("B")
    opt_c = st.text_input("C")
with col2:
    opt_d = st.text_input("D")
    opt_e = st.text_input("E")

if st.button("Solve", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a question prompt.")
    else:
        options = {"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d, "E": opt_e}
        ranked = solver.predict(prompt, options)

        st.subheader("Predicted Ranking")
        top3 = ranked[:3]
        medals = ["🥇", "🥈", "🥉"]
        for medal, (choice, prob) in zip(medals, top3):
            label = options.get(choice, "")
            st.write(f"{medal} **{choice}** — {label}  &nbsp;&nbsp;`{prob:.1%}` confidence")

        with st.expander("Full ranking (all 5 options)"):
            for choice, prob in ranked:
                st.write(f"**{choice}**: {options.get(choice, '')} — `{prob:.1%}`")
