import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.power import TTestIndPower

st.set_page_config(
    page_title="EdTech Experiment & Causal Analysis",
    page_icon="📊",
    layout="wide"
)

# --- AUTO-GENERATE & CLEAN DATA IF MISSING (For Cloud Deployment) ---
@st.cache_data
def load_or_build_data():
    raw_path = "data/raw/edtech_coaching_records.csv"
    weekly_path = "data/processed/cleaned_weekly_records.csv"
    learner_path = "data/processed/learner_level_summary.csv"
    fig_ab = "reports/figures/ab_test_subgroup_cuped_cis.png"
    fig_did = "reports/figures/did_parallel_trends.png"

    if not os.path.exists(learner_path):
        from src.data_generator import generate_edtech_data
        from src.data_cleaner import clean_edtech_data
        generate_edtech_data(output_path=raw_path)
        clean_edtech_data(raw_path=raw_path, weekly_out_path=weekly_path, learner_out_path=learner_path)

    if not (os.path.exists(fig_ab) and os.path.exists(fig_did)):
        from src.ab_test_analyzer import run_ab_test_analysis
        from src.did_analyzer import run_did_analysis
        run_ab_test_analysis(learner_data_path=learner_path, fig_dir="reports/figures")
        run_did_analysis(weekly_data_path=weekly_path, learner_data_path=learner_path, fig_dir="reports/figures")

    learner_df = pd.read_csv(learner_path)
    return learner_df

df = load_or_build_data()

# --- HEADER ---
st.title("📊 EdTech Experimentation & Causal Inference Platform")
st.markdown(
    "**Portfolio Project:** End-to-end telemetry auditing (~400,000 rows), Prospective Power Analysis, "
    "A/B Testing with **CUPED Variance Reduction**, and **Difference-in-Differences (DiD)** for an Ahmedabad coaching chain."
)

# Top KPI Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw Telemetry Audited", "400,000 rows", "-15,000 dups removed")
c2.metric("Q3 A/B Test Lift (CUPED)", "+3.43 pts", "95% CI: [+3.31, +3.55]")
c3.metric("CUPED Variance Reduction", "88.6%", "8x faster recruitment")
c4.metric("Q2 True DiD Lift vs. Naive", "+1.21 pts", "74% of +4.71 pt jump was seasonal")

st.divider()

# --- NAVIGATION TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧹 1. Data Audit (400k Rows)",
    "⚡ 2. Interactive Power Calculator",
    "🧪 3. Q3 A/B Test & CUPED",
    "📈 4. Q2 Diff-in-Diff (DiD)",
    "📋 5. Executive Memo & Validity"
])

# TAB 1: DATA AUDIT
with tab1:
    st.subheader("Telemetry Cleaning Pipeline: Before vs. After")
    st.write("Three real-world instrumentation bugs were remediated before running causal models:")
    
    audit_df = pd.DataFrame({
        "Data Issue / Metric": [
            "1. Total Telemetry Rows",
            "2. Recorded Session Attendance Rate",
            "3. Detected Learner Drop-out Rate",
            "4. Course Completion Rate",
            "5. Mean Score (Overall vs. Active Weeks)"
        ],
        "Raw Buggy Dataset": [
            "400,000 (Includes 15k duplicate rows)",
            "~85% on 30% logged; 70% missing (NaN)",
            "0.0% (All drop-outs still marked 'Active')",
            "100.0% (Artificially inflated by Ghost Actives)",
            f"{df['overall_assessment_score'].mean():.2f} pts (Diluted by post-dropout zeros)"
        ],
        "Cleaned Dataset (48,125 Learners)": [
            "385,000 unique learner-weeks",
            f"{df[df['session_mode']=='Recorded']['attendance_rate'].mean():.1%} (Imputed via assignment/quiz activity)",
            f"{df['dropped_out'].mean():.1%} (Reclassified >=2 trailing zero-activity weeks)",
            f"{df['completed_course'].mean():.1%} (Active in Wk 8 + >=50% attendance)",
            f"{df['active_assessment_score'].mean():.2f} pts (Isolates active academic performance)"
        ]
    })
    st.table(audit_df)
    st.dataframe(df.head(10), use_container_width=True)

# TAB 2: INTERACTIVE POWER ANALYSIS
with tab2:
    st.subheader("Interactive A/B Test Sample Size & Duration Simulator")
    col_in, col_out = st.columns([1, 2])
    
    base_ahm = df[(df['region'] == 'Ahmedabad') & (df['quarter'] == 'Q1_Pre')]
    sd_base = base_ahm['active_assessment_score'].std()
    r_corr = base_ahm['active_assessment_score'].corr(base_ahm['baseline_score'])
    weekly_intake = len(base_ahm) / 8.0

    with col_in:
        mde = st.slider("Target Minimum Detectable Effect (MDE in Points)", 0.5, 4.0, 1.5, 0.1)
        power = st.slider("Statistical Power (1 - β)", 0.70, 0.95, 0.80, 0.05)
        alpha = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1)

    with col_out:
        cohens_d = mde / sd_base
        n_per_arm = int(np.ceil(TTestIndPower().solve_power(effect_size=cohens_d, alpha=alpha, power=power)))
        total_n = n_per_arm * 2
        weeks_needed = total_n / weekly_intake

        cuped_mult = 1.0 - (r_corr ** 2)
        n_cuped_arm = int(np.ceil(n_per_arm * cuped_mult))
        total_cuped = n_cuped_arm * 2
        weeks_cuped = total_cuped / weekly_intake

        m1, m2 = st.columns(2)
        m1.info(f"**Standard Unadjusted T-Test**\n\n"
                f"- Required per arm: **{n_per_arm:,}** learners\n"
                f"- Total Sample Size: **{total_n:,}** learners\n"
                f"- Recruitment Duration: **{weeks_needed:.2f} weeks**")
        m2.success(f"**With CUPED Adjustment (R² = {r_corr**2:.2f})**\n\n"
                   f"- Required per arm: **{n_cuped_arm:,}** learners\n"
                   f"- Total Sample Size: **{total_cuped:,}** learners\n"
                   f"- Recruitment Duration: **{weeks_cuped:.2f} weeks**")

# TAB 3: A/B TEST & CUPED
with tab3:
    st.subheader("Q3 A/B Test Results (Ahmedabad): Sanity Checks, Subgroup Effects & CUPED")
    ab_df = df[(df['quarter'] == 'Q3_AB_Test') & (df['region'] == 'Ahmedabad')].copy()
    
    seg_filter = st.selectbox("Filter by Learner Tier (or view All):", ["All Tiers", "Beginner", "Intermediate", "Advanced"])
    if seg_filter != "All Tiers":
        ab_sub = ab_df[ab_df['tier'] == seg_filter]
    else:
        ab_sub = ab_df

    ctrl = ab_sub[ab_sub['exp_group'] == 'Control']['active_assessment_score']
    trt = ab_sub[ab_sub['exp_group'] == 'Treatment']['active_assessment_score']
    diff = trt.mean() - ctrl.mean()
    se = np.sqrt(ctrl.var()/len(ctrl) + trt.var()/len(trt))

    k1, k2, k3 = st.columns(3)
    k1.metric(f"Control Mean ({seg_filter})", f"{ctrl.mean():.2f} pts", f"N = {len(ctrl):,}")
    k2.metric(f"Treatment Mean ({seg_filter})", f"{trt.mean():.2f} pts", f"N = {len(trt):,}")
    k3.metric("Estimated Lift (ATE)", f"{diff:+.2f} pts", f"95% CI: [{diff-1.96*se:+.2f}, {diff+1.96*se:+.2f}]")

    if os.path.exists("reports/figures/ab_test_subgroup_cuped_cis.png"):
        st.image("reports/figures/ab_test_subgroup_cuped_cis.png", caption="Subgroup Treatment Effects: Unadjusted vs. CUPED 95% CIs")

# TAB 4: DIFFERENCE-IN-DIFFERENCES
with tab4:
    st.subheader("Q2 Unvalidated Launch: Difference-in-Differences (Ahmedabad vs. Vadodara)")
    st.markdown(
        "In `Q2_Launch`, Ahmedabad saw a **+4.71 pt** score increase. By comparing against **Vadodara** (where scores rose **+3.50 pts** due to exam seasonality), "
        "DiD isolates the true product impact of **+1.21 pts** (95% CI: `[+1.06, +1.36]`, Parallel Trends $p = 0.89$)."
    )
    if os.path.exists("reports/figures/did_parallel_trends.png"):
        st.image("reports/figures/did_parallel_trends.png", caption="Weekly Parallel Trends in Q1_Pre and Counterfactual Wedge in Q2_Launch")

# TAB 5: EXECUTIVE MEMO
with tab5:
    if os.path.exists("reports/02_decision_memo.md"):
        with open("reports/02_decision_memo.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())