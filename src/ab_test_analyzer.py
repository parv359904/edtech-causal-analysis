import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

def check_srm(n_control: int, n_treatment: int, expected_ratio: float = 0.50):
    """Chi-square test for Sample Ratio Mismatch (SRM)."""
    total = n_control + n_treatment
    expected = [total * expected_ratio, total * (1.0 - expected_ratio)]
    observed = [n_control, n_treatment]
    chi2_stat, p_val = stats.chisquare(f_obs=observed, f_exp=expected)
    return chi2_stat, p_val

def compute_smd(series_c: pd.Series, series_t: pd.Series) -> float:
    """Computes Standardized Mean Difference (SMD) for covariate balance."""
    mean_c, mean_t = series_c.mean(), series_t.mean()
    var_c, var_t = series_c.var(ddof=1), series_t.var(ddof=1)
    pooled_sd = np.sqrt((var_c + var_t) / 2.0)
    return (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0.0

def welch_ci(series_c: pd.Series, series_t: pd.Series, alpha: float = 0.05):
    """Computes ATE, SE, t-stat, p-value, and 95% CI using Welch's t-test."""
    n_c, n_t = len(series_c), len(series_t)
    mean_c, mean_t = series_c.mean(), series_t.mean()
    var_c, var_t = series_c.var(ddof=1), series_t.var(ddof=1)
    
    ate = mean_t - mean_c
    se = np.sqrt(var_c / n_c + var_t / n_t)
    
    # Welch-Satterthwaite degrees of freedom
    df = (var_c / n_c + var_t / n_t) ** 2 / (
        ((var_c / n_c) ** 2) / (n_c - 1) + ((var_t / n_t) ** 2) / (n_t - 1)
    )
    t_stat = ate / se
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
    t_crit = stats.t.ppf(1 - alpha / 2, df=df)
    ci_low, ci_high = ate - t_crit * se, ate + t_crit * se
    return {
        'control_mean': mean_c,
        'treatment_mean': mean_t,
        'ate': ate,
        'se': se,
        'p_val': p_val,
        'ci_low': ci_low,
        'ci_high': ci_high
    }

def bootstrap_ci(series_c: pd.Series, series_t: pd.Series, n_boot: int = 2000, seed: int = 42):
    """Non-parametric bootstrap 95% Confidence Interval for ATE."""
    rng = np.random.default_rng(seed)
    c_vals = series_c.to_numpy()
    t_vals = series_t.to_numpy()
    
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        c_sample = rng.choice(c_vals, size=len(c_vals), replace=True)
        t_sample = rng.choice(t_vals, size=len(t_vals), replace=True)
        boot_diffs[i] = t_sample.mean() - c_sample.mean()
        
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    return boot_diffs.std(), ci_low, ci_high

def run_ab_test_analysis(
    learner_data_path: str = "data/processed/learner_level_summary.csv",
    fig_dir: str = "reports/figures"
):
    os.makedirs(fig_dir, exist_ok=True)
    df = pd.read_csv(learner_data_path)
    
    # Filter to Q3_AB_Test in Ahmedabad
    ab_df = df[(df['quarter'] == 'Q3_AB_Test') & (df['region'] == 'Ahmedabad')].copy()
    ab_df['is_Beginner'] = (ab_df['tier'] == 'Beginner').astype(int)
    ab_df['is_Recorded'] = (ab_df['session_mode'] == 'Recorded').astype(int)
    
    ctrl = ab_df[ab_df['exp_group'] == 'Control']
    trt = ab_df[ab_df['exp_group'] == 'Treatment']
    
    print("=========================================================================")
    print("       1. SANITY CHECKS: SAMPLE RATIO MISMATCH & COVARIATE BALANCE       ")
    print("=========================================================================")
    chi2_srm, p_srm = check_srm(len(ctrl), len(trt))
    print(f"Sample Size: Control = {len(ctrl):,} ({len(ctrl)/len(ab_df):.2%}), Treatment = {len(trt):,} ({len(trt)/len(ab_df):.2%})")
    print(f"SRM Chi-Square = {chi2_srm:.4f}, p-value = {p_srm:.4f} --> {'PASS (No SRM)' if p_srm >= 0.01 else 'FAIL (SRM Detected!)'}")
    
    print("\nBaseline Covariate Balance (Standardized Mean Difference, threshold |SMD| < 0.10):")
    for cov in ['baseline_score', 'is_Beginner', 'is_Recorded']:
        smd = compute_smd(ctrl[cov], trt[cov])
        _, p_cov = stats.ttest_ind(ctrl[cov], trt[cov], equal_var=False)
        print(f" - {cov:<16}: Control = {ctrl[cov].mean():.3f}, Treatment = {trt[cov].mean():.3f} | SMD = {smd:+.4f} (p = {p_cov:.3f})")

    print("\n=========================================================================")
    print("       2. PRIMARY, SECONDARY & GUARDRAIL METRIC ESTIMATES (95% CI)       ")
    print("=========================================================================")
    metrics = [
        ('active_assessment_score', 'Primary: Active Score (pts)'),
        ('overall_assessment_score', 'Secondary: Overall Score (pts)'),
        ('completed_course', 'Guardrail: Completion Rate'),
        ('dropped_out', 'Guardrail: Drop-off Rate'),
        ('missing_attendance_rate', 'Guardrail: Missing Rec. Logs')
    ]
    
    for col, label in metrics:
        res = welch_ci(ctrl[col], trt[col])
        if col == 'active_assessment_score':
            boot_se, b_low, b_high = bootstrap_ci(ctrl[col], trt[col])
            print(f"{label:<31} | Ctrl: {res['control_mean']:.2f}, Trt: {res['treatment_mean']:.2f} | "
                  f"ATE: {res['ate']:+.3f} (Welch 95% CI: [{res['ci_low']:+.3f}, {res['ci_high']:+.3f}], "
                  f"Boot 95% CI: [{b_low:+.3f}, {b_high:+.3f}], p = {res['p_val']:.2e})")
        else:
            print(f"{label:<31} | Ctrl: {res['control_mean']:.3f}, Trt: {res['treatment_mean']:.3f} | "
                  f"ATE: {res['ate']:+.4f} (95% CI: [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}], p = {res['p_val']:.2e})")

    print("\n=========================================================================")
    print("       3. STRETCH: CUPED & REGRESSION ADJUSTMENT VARIANCE REDUCTION      ")
    print("=========================================================================")
    # Compute CUPED adjusted outcome using pre-experiment baseline_score
    cov_matrix = np.cov(ab_df['active_assessment_score'], ab_df['baseline_score'], ddof=1)
    theta = cov_matrix[0, 1] / cov_matrix[1, 1]
    mean_baseline = ab_df['baseline_score'].mean()
    
    ab_df['cuped_active_score'] = ab_df['active_assessment_score'] - theta * (ab_df['baseline_score'] - mean_baseline)
    
    raw_res = welch_ci(ctrl['active_assessment_score'], trt['active_assessment_score'])
    cuped_res = welch_ci(
        ab_df[ab_df['exp_group'] == 'Control']['cuped_active_score'],
        ab_df[ab_df['exp_group'] == 'Treatment']['cuped_active_score']
    )
    
    # Also verify with OLS Regression Adjustment controlling for baseline_score + tier + session_mode
    ols_model = smf.ols("active_assessment_score ~ C(exp_group, Treatment(reference='Control')) + baseline_score + C(tier) + C(session_mode)", data=ab_df).fit(cov_type='HC3')
    ols_ate = ols_model.params["C(exp_group, Treatment(reference='Control'))[T.Treatment]"]
    ols_se = ols_model.bse["C(exp_group, Treatment(reference='Control'))[T.Treatment]"]
    ols_ci = ols_model.conf_int().loc["C(exp_group, Treatment(reference='Control'))[T.Treatment]"]

    var_reduction_pct = (1.0 - (cuped_res['se'] ** 2) / (raw_res['se'] ** 2)) * 100.0
    print(f"CUPED Theta (Cov(Y,X)/Var(X)): {theta:.4f}")
    print(f" - Unadjusted Welch Estimate:  ATE = {raw_res['ate']:+.3f} pts | SE = {raw_res['se']:.4f} | 95% CI = [{raw_res['ci_low']:+.3f}, {raw_res['ci_high']:+.3f}]")
    print(f" - CUPED Adjusted Estimate:    ATE = {cuped_res['ate']:+.3f} pts | SE = {cuped_res['se']:.4f} | 95% CI = [{cuped_res['ci_low']:+.3f}, {cuped_res['ci_high']:+.3f}]")
    print(f" - OLS Covariate-Adjusted:     ATE = {ols_ate:+.3f} pts | SE = {ols_se:.4f} | 95% CI = [{ols_ci[0]:+.3f}, {ols_ci[1]:+.3f}]")
    print(f" --> Variance Reduced by:      {var_reduction_pct:.1f}% (Test recruitment could have been {var_reduction_pct:.1f}% shorter!)")

    print("\n=========================================================================")
    print("       4. SUBGROUP SEGMENTATION (HETEROGENEOUS TREATMENT EFFECTS)        ")
    print("=========================================================================")
    seg_records = []
    for seg_col in ['tier', 'session_mode']:
        for val in sorted(ab_df[seg_col].unique()):
            sub_c = ctrl[ctrl[seg_col] == val]['active_assessment_score']
            sub_t = trt[trt[seg_col] == val]['active_assessment_score']
            # Use CUPED score for tighter subgroup CIs alongside raw score
            sub_c_cuped = ab_df[(ab_df['exp_group'] == 'Control') & (ab_df[seg_col] == val)]['cuped_active_score']
            sub_t_cuped = ab_df[(ab_df['exp_group'] == 'Treatment') & (ab_df[seg_col] == val)]['cuped_active_score']
            
            r_raw = welch_ci(sub_c, sub_t)
            r_cup = welch_ci(sub_c_cuped, sub_t_cuped)
            seg_records.append({
                'Segment': f"{seg_col}: {val}",
                'N': len(sub_c) + len(sub_t),
                'Raw_ATE': r_raw['ate'],
                'Raw_CI_Low': r_raw['ci_low'],
                'Raw_CI_High': r_raw['ci_high'],
                'CUPED_ATE': r_cup['ate'],
                'CUPED_CI_Low': r_cup['ci_low'],
                'CUPED_CI_High': r_cup['ci_high'],
                'p_val': r_cup['p_val']
            })
            print(f" - {seg_col:<12} = {val:<12} (N={len(sub_c)+len(sub_t):,}) | "
                  f"Raw ATE: {r_raw['ate']:+.2f} [{r_raw['ci_low']:+.2f}, {r_raw['ci_high']:+.2f}] | "
                  f"CUPED ATE: {r_cup['ate']:+.2f} [{r_cup['ci_low']:+.2f}, {r_cup['ci_high']:+.2f}] (p={r_cup['p_val']:.2e})")

    # Save visual plot of Subgroup & CUPED Confidence Intervals
    seg_df = pd.DataFrame(seg_records)
    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(seg_df))
    
    ax.errorbar(seg_df['Raw_ATE'], y_pos + 0.15,
                xerr=[seg_df['Raw_ATE'] - seg_df['Raw_CI_Low'], seg_df['Raw_CI_High'] - seg_df['Raw_ATE']],
                fmt='o', color='gray', label='Unadjusted 95% CI', capsize=4)
    ax.errorbar(seg_df['CUPED_ATE'], y_pos - 0.15,
                xerr=[seg_df['CUPED_ATE'] - seg_df['CUPED_CI_Low'], seg_df['CUPED_CI_High'] - seg_df['CUPED_ATE']],
                fmt='s', color='#1f77b4', label='CUPED Adjusted 95% CI', capsize=4)
    
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(seg_df['Segment'])
    ax.set_xlabel('Treatment Effect on Active Assessment Score (Points)')
    ax.set_title('A/B Test Subgroup Treatment Effects: Unadjusted vs. CUPED 95% CIs')
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(fig_dir, "ab_test_subgroup_cuped_cis.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved Subgroup & CUPED CI Forest Plot to '{plot_path}'!")

if __name__ == "__main__":
    run_ab_test_analysis()