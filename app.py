from pathlib import Path

import pandas as pd
import streamlit as st

st.title("Insurance Claims Review")
st.caption("Explore review capacity using the model’s validation results.")

scenario_file = Path(__file__).with_name("validation_scenarios.csv")
scenarios = pd.read_csv(scenario_file).set_index("Review capacity")

capacity = st.select_slider(
    "Review capacity",
    options=scenarios.index.tolist(),
    value="20%",
)
selected = scenarios.loc[capacity]

left, middle, right = st.columns(3)
left.metric("Claims referred", f"{selected['Claims referred']:,}")
middle.metric("Fraud recall", selected["Fraud recall"])
right.metric("Referral precision", selected["Referral precision"])

st.write(
    f"At a score cutoff of **{selected['Score cutoff']:.4f}**, "
    f"the review queue includes **{selected['Fraud cases caught']:,}** "
    "claims labelled as fraud."
)

st.dataframe(scenarios.reset_index())

st.header("Held-out test results")
st.caption("Results on 5,496 later claims using the cutoff selected at 20% validation capacity.")

review, recall, precision = st.columns(3)
review.metric("Claims referred", "19.5%")
recall.metric("Fraud recall", "30.3%")
precision.metric("Referral precision", "32.3%")

st.write(
    "The model referred 1,070 claims for review and caught 346 of the "
    "1,141 claims labelled as fraud. It missed the other 795."
)