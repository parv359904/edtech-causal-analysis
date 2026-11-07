import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

def run_did_analysis(
    weekly_data_path: str = "data/processed/cleaned_weekly_records.csv",
    learner_data_path: str = "data/processed/learner_level_summary.csv",
    fig_dir: str = "reports/figures"
):
    """
    1. Tests the Parallel Trends assumption in Q1_Pre across Weeks 1-8.
    2. Estimates the Difference-in-Differences (DiD) causal lift in Q2_Launch
       between Ahmedabad (Treatment Region) and Vadodara (Control Region).
    3. Segments the DiD effect across learner tiers and session modes.
    """
    os.makedirs(fig_dir, exist_ok=True)
    weekly_df = pd.read_csv(weekly_data_path)
    learner_df = pd.read_csv(learner_data_path)

    # Filter to Q1_Pre and Q2_Launch (the observational DiD window)
    did_learners = learner_df[learner_df['quarter'].isin(['Q1_Pre', 'Q2_Launch'])].copy()
    did_learners['is_ahmedabad'] = (did_learners['region'] == 'Ahmedabad').astype(int)
    did_learners['is_post_q2'] = (did_learners['quarter'] == 'Q2_Launch').astype(int)

    # =========================================================================
    # 1. PARALLEL TRENDS TEST (Pre-Intervention Q1_Pre Weekly Trajectory)
    # =========================================================================
    print("=========================================================================")
    print("       1. PARALLEL TRENDS ASSUMPTION CHECK (PRE-PERIOD: Q1_Pre)          ")
    print("=========================================================================")
    q1_weekly = weekly_df[
        (weekly_df['quarter'] == 'Q1_Pre') &
        (weekly_df['enrolled_status_clean'] == 'Active')
    ].copy()
    q1_weekly['is_ahmedabad'] = (q1_weekly['region'] == 'Ahmedabad').astype(int)

    # Test if weekly slope differs between Ahmedabad and Vadodara in Q1_Pre
    pt_model = smf.ols(
        "assessment_score ~ week * is_ahmedabad",
        data=q1_weekly
    ).fit(cov_type='cluster', cov_kwds={'groups': q1_weekly['batch_id']})

    slope_diff = pt_model.params['week:is_ahmedabad']
    slope_p = pt_model.pvalues['week:is_ahmedabad']
    slope_ci = pt_model.conf_int().loc['week:is_ahmedabad']

    print(f"Pre-Trend Slope Difference (Ahmedabad vs. Vadodara per week): {slope_diff:+.4f} pts/week")
    print(f"95% CI: [{slope_ci[0]:+.4f}, {slope_ci[1]:+.4f}] | p-value = {slope_p:.4f}")
    print(f"Status: {'PASS (Parallel Trends Hold - No diverging pre-trend)' if slope_p > 0.05 else 'FAIL (Pre-trends diverge!)'}")

    # =========================================================================
    # 2. DIFFERENCE-IN-DIFFERENCES (DiD) ESTIMATION (Q1_Pre vs. Q2_Launch)
    # =========================================================================
    print("\n=========================================================================")
    print("       2. DIFFERENCE-IN-DIFFERENCES (DiD) IMPACT OF Q2 LAUNCH            ")
    print("=========================================================================")
    # 2x2 Cell Means
    means = did_learners.groupby(['region', 'quarter'])['active_assessment_score'].mean().unstack()
    ahm_diff = means.loc['Ahmedabad', 'Q2_Launch'] - means.loc['Ahmedabad', 'Q1_Pre']
    vad_diff = means.loc['Vadodara', 'Q2_Launch'] - means.loc['Vadodara', 'Q1_Pre']
    raw_did = ahm_diff - vad_diff

    print(f"Ahmedabad Mean Score: Q1_Pre = {means.loc['Ahmedabad', 'Q1_Pre']:.2f} -> Q2_Launch = {means.loc['Ahmedabad', 'Q2_Launch']:.2f} (Naive Change = {ahm_diff:+.2f} pts)")
    print(f"Vadodara Mean Score:  Q1_Pre = {means.loc['Vadodara', 'Q1_Pre']:.2f} -> Q2_Launch = {means.loc['Vadodara', 'Q2_Launch']:.2f} (Seasonal Change = {vad_diff:+.2f} pts)")
    print(f"Naive Claim vs. Reality: Out of the {ahm_diff:+.2f} pt jump in Ahmedabad, {vad_diff:+.2f} pts ({vad_diff/ahm_diff:.1%}) was just seasonality!")

    # Unadjusted OLS DiD (Cluster-robust SE by batch_id)
    did_unadj = smf.ols(
        "active_assessment_score ~ is_ahmedabad * is_post_q2",
        data=did_learners
    ).fit(cov_type='cluster', cov_kwds={'groups': did_learners['batch_id']})

    # Covariate-Adjusted OLS DiD (Controlling for baseline_score, tier, session_mode)
    did_adj = smf.ols(
        "active_assessment_score ~ is_ahmedabad * is_post_q2 + baseline_score + C(tier) + C(session_mode)",
        data=did_learners
    ).fit(cov_type='cluster', cov_kwds={'groups': did_learners['batch_id']})

    for name, model in [("Unadjusted DiD (Cluster-Robust)", did_unadj), ("Covariate-Adjusted DiD (CUPED-style)", did_adj)]:
        est = model.params['is_ahmedabad:is_post_q2']
        se = model.bse['is_ahmedabad:is_post_q2']
        ci = model.conf_int().loc['is_ahmedabad:is_post_q2']
        pval = model.pvalues['is_ahmedabad:is_post_q2']
        print(f" - {name:<36}: Lift = {est:+.3f} pts | SE = {se:.4f} | 95% CI = [{ci[0]:+.3f}, {ci[1]:+.3f}] (p = {pval:.2e})")

    # Check Guardrail Metric (Drop-off Rate) in DiD
    did_drop = smf.ols(
        "dropped_out ~ is_ahmedabad * is_post_q2 + baseline_score",
        data=did_learners
    ).fit(cov_type='cluster', cov_kwds={'groups': did_learners['batch_id']})
    drop_est = did_drop.params['is_ahmedabad:is_post_q2']
    drop_ci = did_drop.conf_int().loc['is_ahmedabad:is_post_q2']
    drop_p = did_drop.pvalues['is_ahmedabad:is_post_q2']
    print(f" - Guardrail (Drop-off Rate DiD)       : Lift = {drop_est:+.4f} | 95% CI = [{drop_ci[0]:+.4f}, {drop_ci[1]:+.4f}] (p = {drop_p:.3f})")

    # =========================================================================
    # 3. SEGMENTATION OF Q2 DiD EFFECT
    # =========================================================================
    print("\n=========================================================================")
    print("       3. SUBGROUP SEGMENTATION OF Q2 DiD EFFECT                         ")
    print("=========================================================================")
    for seg_col in ['tier', 'session_mode']:
        for val in sorted(did_learners[seg_col].unique()):
            sub_df = did_learners[did_learners[seg_col] == val]
            sub_mod = smf.ols(
                "active_assessment_score ~ is_ahmedabad * is_post_q2 + baseline_score",
                data=sub_df
            ).fit(cov_type='cluster', cov_kwds={'groups': sub_df['batch_id']})
            est = sub_mod.params['is_ahmedabad:is_post_q2']
            ci = sub_mod.conf_int().loc['is_ahmedabad:is_post_q2']
            pval = sub_mod.pvalues['is_ahmedabad:is_post_q2']
            print(f" - {seg_col:<12} = {val:<12} (N={len(sub_df):,}) | DiD Lift: {est:+.2f} pts (95% CI: [{ci[0]:+.2f}, {ci[1]:+.2f}], p={pval:.3e})")

    # =========================================================================
    # 4. PLOT PARALLEL TRENDS & DiD TRAJECTORY across Q1_Pre and Q2_Launch
    # =========================================================================
    did_weekly = weekly_df[
        (weekly_df['quarter'].isin(['Q1_Pre', 'Q2_Launch'])) &
        (weekly_df['enrolled_status_clean'] == 'Active')
    ].copy()
    did_weekly['continuous_week'] = np.where(
        did_weekly['quarter'] == 'Q1_Pre',
        did_weekly['week'],
        did_weekly['week'] + 8
    )

    trend_summary = did_weekly.groupby(['region', 'continuous_week'])['assessment_score'].agg(
        mean='mean',
        se=lambda x: x.std(ddof=1) / np.sqrt(len(x))
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {'Ahmedabad': '#d62728', 'Vadodara': '#1f77b4'}

    for reg in ['Ahmedabad', 'Vadodara']:
        sub = trend_summary[trend_summary['region'] == reg]
        ax.plot(sub['continuous_week'], sub['mean'], marker='o', label=f"{reg}", color=colors[reg], linewidth=2)
        ax.fill_between(sub['continuous_week'], sub['mean'] - 1.96 * sub['se'], sub['mean'] + 1.96 * sub['se'],
                        color=colors[reg], alpha=0.15)

    # Counterfactual line for Ahmedabad in Q2
    q1_ahm_mean = trend_summary[(trend_summary['region'] == 'Ahmedabad') & (trend_summary['continuous_week'] <= 8)]['mean'].mean()
    q1_vad_mean = trend_summary[(trend_summary['region'] == 'Vadodara') & (trend_summary['continuous_week'] <= 8)]['mean'].mean()
    vad_q2 = trend_summary[(trend_summary['region'] == 'Vadodara') & (trend_summary['continuous_week'] > 8)]
    ax.plot(vad_q2['continuous_week'], vad_q2['mean'] + (q1_ahm_mean - q1_vad_mean),
            linestyle='--', color='#d62728', alpha=0.7, label='Ahmedabad Counterfactual (No Q2 Change)')

    ax.axvline(8.5, color='black', linestyle=':', linewidth=1.5, label='Q2 Launch Intervention Boundary')
    ax.set_xticks(range(1, 17))
    ax.set_xticklabels([f"Q1-W{w}" for w in range(1, 9)] + [f"Q2-W{w}" for w in range(1, 9)], rotation=45)
    ax.set_ylabel("Active Assessment Score (Points)")
    ax.set_title("Difference-in-Differences: Weekly Parallel Trends (Q1_Pre) & Post-Launch Impact (Q2_Launch)")
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(fig_dir, "did_parallel_trends.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved Parallel Trends & DiD Chart to '{plot_path}'!")

if __name__ == "__main__":
    run_did_analysis()