import os
import numpy as np
import pandas as pd

def clean_edtech_data(
    raw_path: str = "data/raw/edtech_coaching_records.csv",
    weekly_out_path: str = "data/processed/cleaned_weekly_records.csv",
    learner_out_path: str = "data/processed/learner_level_summary.csv"
):
    print(f"Loading raw data from '{raw_path}'...")
    df = pd.read_csv(raw_path)
    raw_row_count = len(df)

    # ---------------------------------------------------------
    # FIX 1: Deduplicate (learner_id, batch_id, week)
    # ---------------------------------------------------------
    df = df.sort_values(['learner_id', 'batch_id', 'week']).reset_index(drop=True)
    df_dedup = df.drop_duplicates(subset=['learner_id', 'batch_id', 'week'], keep='first').copy()
    dups_removed = raw_row_count - len(df_dedup)
    print(f"[Fix 1] Removed {dups_removed:,} duplicate enrolment/weekly rows. Remaining: {len(df_dedup):,}")

    # ---------------------------------------------------------
    # FIX 2: Impute Missing Attendance for Recorded Sessions
    # ---------------------------------------------------------
    missing_att_before = df_dedup['attended'].isna().sum()
    
    # Flag whether attendance was originally missing (useful as a guardrail metric)
    df_dedup['attendance_was_missing'] = df_dedup['attended'].isna().astype(int)

    # Rule: If Recorded session attendance is NaN, infer 1.0 if assignment submitted or score > 0, else 0.0
    inferred_attendance = np.where(
        (df_dedup['assignment_submitted'] == 1) | (df_dedup['assessment_score'] > 0),
        1.0,
        0.0
    )
    df_dedup['attended'] = np.where(
        df_dedup['attended'].isna(),
        inferred_attendance,
        df_dedup['attended']
    )
    print(f"[Fix 2] Imputed {missing_att_before:,} missing 'Recorded' attendance logs using assignment/assessment activity.")

    # ---------------------------------------------------------
    # FIX 3: Identify "Ghost Active" Learners (True Drop-outs)
    # ---------------------------------------------------------
    # A week is completely inactive if attended == 0, assignment_submitted == 0, and assessment_score == 0
    df_dedup['zero_activity_week'] = (
        (df_dedup['attended'] == 0) &
        (df_dedup['assignment_submitted'] == 0) &
        (df_dedup['assessment_score'] == 0)
    ).astype(int)

    # Sort reverse by week per learner to find trailing consecutive inactive weeks at the end of the course
    df_dedup = df_dedup.sort_values(['learner_id', 'week'], ascending=[True, False])
    
    # Cumulative product of zero_activity_week from Week 8 backwards identifies uninterrupted trailing inactivity
    df_dedup['trailing_inactive'] = df_dedup.groupby('learner_id')['zero_activity_week'].cumprod()
    
    # Restore chronological order
    df_dedup = df_dedup.sort_values(['learner_id', 'week'], ascending=[True, True]).reset_index(drop=True)

    # Count how many trailing inactive weeks each learner has at the end of the 8-week quarter
    trailing_counts = df_dedup.groupby('learner_id')['trailing_inactive'].transform('sum')

    # Rule: If a learner has >= 2 consecutive inactive weeks through Week 8, they dropped out
    df_dedup['enrolled_status_clean'] = np.where(
        (df_dedup['trailing_inactive'] == 1) & (trailing_counts >= 2),
        'Dropped_Out',
        'Active'
    )

    ghost_weeks_fixed = (df_dedup['enrolled_status_clean'] == 'Dropped_Out').sum()
    print(f"[Fix 3] Reclassified {ghost_weeks_fixed:,} ghost 'Active' learner-weeks as 'Dropped_Out'.")

    # ---------------------------------------------------------
    # BUILD LEARNER-LEVEL SUMMARY (1 Row = 1 Unique Learner)
    # ---------------------------------------------------------
    print("Aggregating to learner-level summary (48,125 unique learners)...")
    
    # Separate active weeks score so dropouts' trailing 0s don't distort active academic performance
    df_dedup['active_week_score'] = np.where(
        df_dedup['enrolled_status_clean'] == 'Active',
        df_dedup['assessment_score'],
        np.nan
    )

    learner_df = df_dedup.groupby(
        ['learner_id', 'region', 'quarter', 'batch_id', 'tier', 'session_mode', 'exp_group', 'baseline_score'],
        as_index=False
    ).agg(
        weeks_active=('enrolled_status_clean', lambda x: (x == 'Active').sum()),
        attendance_rate=('attended', 'mean'),
        assignment_completion_rate=('assignment_submitted', 'mean'),
        overall_assessment_score=('assessment_score', 'mean'),
        active_assessment_score=('active_week_score', 'mean'),
        missing_attendance_rate=('attendance_was_missing', 'mean')
    )

    # Define binary Course Completion & Drop-off indicators
    learner_df['dropped_out'] = (learner_df['weeks_active'] < 8).astype(int)
    learner_df['completed_course'] = (
        (learner_df['dropped_out'] == 0) & (learner_df['attendance_rate'] >= 0.50)
    ).astype(int)

    # Save processed datasets
    os.makedirs(os.path.dirname(weekly_out_path), exist_ok=True)
    df_dedup.drop(columns=['zero_activity_week', 'trailing_inactive', 'active_week_score'], inplace=True)
    df_dedup.to_csv(weekly_out_path, index=False)
    learner_df.to_csv(learner_out_path, index=False)

    print("\n--- BEFORE vs. AFTER CLEANING SUMMARY ---")
    print(f"Total Rows:                  {raw_row_count:,} (Raw)  -->  {len(df_dedup):,} (Cleaned Weekly) / {len(learner_df):,} (Unique Learners)")
    print(f"Recorded Attendance Rate:    {df[df['session_mode']=='Recorded']['attended'].mean():.1%} (Raw, ignoring NaNs)  -->  {learner_df[learner_df['session_mode']=='Recorded']['attendance_rate'].mean():.1%} (Imputed)")
    print(f"Detected Drop-out Rate:      0.0% (Raw Bug)  -->  {learner_df['dropped_out'].mean():.1%} (Cleaned)")
    print(f"Course Completion Rate:      100.0% (Raw Bug) -->  {learner_df['completed_course'].mean():.1%} (Cleaned)")
    print(f"Mean Score (With Dropouts):  {learner_df['overall_assessment_score'].mean():.2f}")
    print(f"Mean Score (Active Weeks):   {learner_df['active_assessment_score'].mean():.2f}")
    print(f"\nSaved cleaned files to '{weekly_out_path}' and '{learner_out_path}'!")

if __name__ == "__main__":
    clean_edtech_data()