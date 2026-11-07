import os
import numpy as np
import pandas as pd

def generate_edtech_data(output_path: str = "data/raw/edtech_coaching_records.csv", seed: int = 42):
    """
    Generates a 400,000-row synthetic EdTech dataset for Ahmedabad & Vadodara
    across 3 quarters (Q1_Pre, Q2_Launch, Q3_AB_Test) with 3 realistic data bugs.
    """
    np.random.seed(seed)
    print("1. Generating base learner profiles (48,125 learners)...")

    n_learners = 48125
    weeks = np.arange(1, 9)  # 8 weeks per quarter

    learners = pd.DataFrame({
        'learner_id': [f'LRN_{i:06d}' for i in range(1, n_learners + 1)],
        'region': np.random.choice(['Ahmedabad', 'Vadodara'], size=n_learners, p=[0.60, 0.40]),
        'quarter': np.random.choice(['Q1_Pre', 'Q2_Launch', 'Q3_AB_Test'], size=n_learners, p=[0.30, 0.35, 0.35]),
        'baseline_score': np.clip(np.random.normal(62, 12, n_learners), 20, 98).round(2),
        'tier': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], size=n_learners, p=[0.35, 0.45, 0.20]),
        'session_mode': np.random.choice(['Live', 'Recorded'], size=n_learners, p=[0.65, 0.35])
    })

    # Assign batch_id and A/B test group (A/B test only runs in Q3_AB_Test in Ahmedabad)
    learners['batch_id'] = (
        learners['region'].str[:3].str.upper() + '_B' +
        np.random.randint(101, 160, size=n_learners).astype(str)
    )
    
    is_q3_ahm = (learners['quarter'] == 'Q3_AB_Test') & (learners['region'] == 'Ahmedabad')
    learners['exp_group'] = np.where(
        is_q3_ahm,
        np.random.choice(['Control', 'Treatment'], size=n_learners, p=[0.50, 0.50]),
        'Not_In_Exp'
    )

    # Simulate true dropout week at the learner level (~18% of learners drop out between week 3 and 7)
    # Treatment in Q3 slightly reduces dropout probability (from 18% to 14%)
    dropout_prob = np.where(learners['exp_group'] == 'Treatment', 0.14, 0.18)
    learners['true_dropout_week'] = np.where(
        np.random.rand(n_learners) < dropout_prob,
        np.random.randint(3, 8, size=n_learners),
        99  # 99 means never dropped out
    )

    print("2. Expanding to 8 weekly logs per learner (385,000 unique rows)...")
    df = learners.loc[learners.index.repeat(len(weeks))].copy()
    df['week'] = np.tile(weeks, n_learners)

    # Check if learner has dropped out by the current week
    is_dropped = df['week'] >= df['true_dropout_week']

    # --- SIMULATE CAUSAL EFFECTS ---
    # Seasonality: Everyone in Q2_Launch gets +3.5 points
    seasonal_effect = np.where(df['quarter'] == 'Q2_Launch', 3.5, 0.0)

    # DiD True Effect: Ahmedabad in Q2_Launch gets +1.2 points
    did_effect = np.where((df['quarter'] == 'Q2_Launch') & (df['region'] == 'Ahmedabad'), 1.2, 0.0)

    # A/B Test True Effect: Treatment in Q3_AB_Test gets +2.8 base, plus +1.8 extra for Beginners
    ab_effect = np.where(df['exp_group'] == 'Treatment', 2.8, 0.0)
    ab_effect += np.where((df['exp_group'] == 'Treatment') & (df['tier'] == 'Beginner'), 1.8, 0.0)

    # Generate weekly activity & outcomes
    df['assignment_submitted'] = np.where(is_dropped, 0, np.random.binomial(1, 0.82, len(df)))
    df['attended'] = np.where(is_dropped, 0.0, np.random.binomial(1, 0.85, len(df)).astype(float))

    raw_score = (
        0.75 * df['baseline_score'] + 15.0 +
        seasonal_effect + did_effect + ab_effect +
        np.random.normal(0, 6.0, len(df))
    )
    df['assessment_score'] = np.where(is_dropped, 0.0, np.clip(raw_score, 0, 100).round(2))

    # Drop the internal helper column so our cleaner has to discover dropouts from the data
    df.drop(columns=['true_dropout_week'], inplace=True)

    print("3. Injecting the 3 real-world logging bugs...")
    # BUG 1: Drop-outs are still marked 'Active' in the database
    df['enrolled_status'] = 'Active'

    # BUG 2: 70% of Recorded session rows have missing (NaN) attendance
    recorded_mask = (df['session_mode'] == 'Recorded') & (np.random.rand(len(df)) < 0.70)
    df.loc[recorded_mask, 'attended'] = np.nan

    # BUG 3: 15,000 duplicate enrolment/log rows to hit 400,000 rows
    duplicates = df.sample(n=15000, random_state=seed).copy()
    df_final = pd.concat([df, duplicates], ignore_index=True)

    # Shuffle rows so duplicates aren't all at the bottom
    df_final = df_final.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Success! Saved {len(df_final):,} rows to '{output_path}'.")
    print("\nQuick Audit of Raw Data Bugs:")
    print(f" - Total Rows: {len(df_final):,}")
    print(f" - Duplicate (learner_id, batch_id, week) rows: {df_final.duplicated(subset=['learner_id', 'batch_id', 'week']).sum():,}")
    print(f" - Missing 'attended' values: {df_final['attended'].isna().sum():,}")
    print(f" - Learners marked 'Active' despite 0 score & 0 assignment: {((df_final['enrolled_status'] == 'Active') & (df_final['assessment_score'] == 0) & (df_final['assignment_submitted'] == 0)).sum():,}")

if __name__ == "__main__":
    generate_edtech_data()