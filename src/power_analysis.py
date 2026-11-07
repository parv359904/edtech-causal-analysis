import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

def run_power_analysis(
    learner_data_path: str = "data/processed/learner_level_summary.csv",
    mde_score: float = 1.50,
    mde_completion_pp: float = 0.03,
    alpha: float = 0.05,
    power: float = 0.80
):
    """
    Calculates baseline metrics from historical Q1_Pre data in Ahmedabad
    and computes required sample size and test duration for the Q3 A/B test.
    """
    df = pd.read_csv(learner_data_path)
    
    # Use historical pre-intervention Ahmedabad data (Q1_Pre) as our baseline
    base_df = df[(df['region'] == 'Ahmedabad') & (df['quarter'] == 'Q1_Pre')].copy()
    
    # 1. Baseline statistics
    n_historical = len(base_df)
    weekly_enrolment_rate = n_historical / 8.0  # 8-week quarter enrolment velocity
    
    mean_active_score = base_df['active_assessment_score'].mean()
    sd_active_score = base_df['active_assessment_score'].std()
    
    mean_overall_score = base_df['overall_assessment_score'].mean()
    sd_overall_score = base_df['overall_assessment_score'].std()
    
    base_completion_rate = base_df['completed_course'].mean()
    base_dropout_rate = base_df['dropped_out'].mean()

    # 2. Sample Size for Primary Metric (Continuous: Active Assessment Score)
    cohens_d = mde_score / sd_active_score
    power_calc = TTestIndPower()
    n_per_arm_score = int(np.ceil(
        power_calc.solve_power(effect_size=cohens_d, alpha=alpha, power=power, ratio=1.0)
    ))
    total_n_score = n_per_arm_score * 2
    weeks_needed_score = total_n_score / weekly_enrolment_rate

    # 3. Sample Size for Guardrail/Alternative Binary Metric (Course Completion Rate: +3% lift)
    effect_size_prop = proportion_effectsize(base_completion_rate + mde_completion_pp, base_completion_rate)
    prop_power_calc = NormalIndPower()
    n_per_arm_comp = int(np.ceil(
        prop_power_calc.solve_power(effect_size=effect_size_prop, alpha=alpha, power=power, ratio=1.0)
    ))
    total_n_comp = n_per_arm_comp * 2
    weeks_needed_comp = total_n_comp / weekly_enrolment_rate

    # 4. Stretch Preview: How much sample size CUPED saves using baseline_score
    r_corr = base_df['active_assessment_score'].corr(base_df['baseline_score'])
    cuped_variance_multiplier = 1.0 - (r_corr ** 2)
    n_per_arm_cuped = int(np.ceil(n_per_arm_score * cuped_variance_multiplier))
    total_n_cuped = n_per_arm_cuped * 2
    weeks_needed_cuped = total_n_cuped / weekly_enrolment_rate

    print("=================================================================")
    print("       A/B TEST POWER ANALYSIS & DURATION REPORT (AHMEDABAD)     ")
    print("=================================================================")
    print(f"Historical Baseline (Q1_Pre Ahmedabad): N = {n_historical:,} learners (~{weekly_enrolment_rate:.0f} learners/week)")
    print(f" - Active Assessment Score:   Mean = {mean_active_score:.2f}, SD = {sd_active_score:.2f}")
    print(f" - Overall Assessment Score:  Mean = {mean_overall_score:.2f}, SD = {sd_overall_score:.2f}")
    print(f" - Course Completion Rate:    {base_completion_rate:.2%}")
    print(f" - Learner Drop-off Rate:     {base_dropout_rate:.2%}")
    print("-----------------------------------------------------------------")
    print(f"1. PRIMARY METRIC: Active Assessment Score (Target MDE = +{mde_score:.2f} pts)")
    print(f" - Standardized Effect (Cohen's d): {cohens_d:.4f}")
    print(f" - Required Sample Size:            {n_per_arm_score:,} per arm ({total_n_score:,} total learners)")
    print(f" - Recruitment Duration:            {weeks_needed_score:.2f} weeks (+ 8-week course window)")
    print("-----------------------------------------------------------------")
    print(f"2. GUARDRAIL BINARY METRIC: Course Completion Rate (Target MDE = +{mde_completion_pp*100:.1f}% pp)")
    print(f" - Required Sample Size:            {n_per_arm_comp:,} per arm ({total_n_comp:,} total learners)")
    print(f" - Recruitment Duration:            {weeks_needed_comp:.2f} weeks")
    print("-----------------------------------------------------------------")
    print(f"3. STRETCH (CUPED VARIANCE REDUCTION PREVIEW):")
    print(f" - Pre-Test Correlation (r):        {r_corr:.4f} (Variance reduced by {(r_corr**2)*100:.1f}%)")
    print(f" - CUPED Required Sample Size:      {n_per_arm_cuped:,} per arm ({total_n_cuped:,} total learners)")
    print(f" - CUPED Recruitment Duration:      {weeks_needed_cuped:.2f} weeks (Saves {weeks_needed_score - weeks_needed_cuped:.2f} weeks!)")
    print("=================================================================")

if __name__ == "__main__":
    run_power_analysis()