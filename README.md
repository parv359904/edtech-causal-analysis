# EdTech Experiment & Causal Analysis

End-to-end experimentation and causal inference pipeline evaluating pedagogical changes for a multi-centre coaching institute in Ahmedabad (~400,000 raw learner-week logs across 48,125 learners).

## Key Objectives Covered
1. **Telemetry Auditing & Data Cleaning (`src/data_cleaner.py`):**
   - Deduplicated 15,000 double-enrolled `(learner_id, batch_id, week)` records.
   - Reclassified 17.3% "Ghost Active" drop-outs exhibiting trailing multi-week inactivity.
   - Imputed 94,000+ missing (`NaN`) attendance records for `Recorded` sessions via verified assessment/assignment activity.
2. **Prospective A/B Test Design & Power Analysis (`src/power_analysis.py` & `reports/01_experiment_design_doc.md`):**
   - Established primary (`active_assessment_score`) and guardrail (`dropped_out`, `completed_course`) metrics, computing exact sample size and recruitment duration for an MDE of +1.50 points ($\alpha=0.05$, Power $=0.80$).
3. **RCT Evaluation & Variance Reduction (`src/ab_test_analyzer.py`):**
   - Verified randomization health via Sample Ratio Mismatch ($\chi^2$ test) and covariate Standardized Mean Differences ($|\text{SMD}| < 0.02$).
   - Estimated treatment effects using Welch's t-test and 2,000-iteration non-parametric bootstrapping.
   - Applied **CUPED** using pre-enrolment `baseline_score`, reducing sample variance by **88.6%**.
4. **Observational Causal Inference (`src/did_analyzer.py`):**
   - Evaluated an unvalidated Q2 launch in Ahmedabad against Vadodara using **Difference-in-Differences (DiD)** with batch-clustered standard errors, separating a +3.50 pt seasonal surge from the true +1.21 pt product lift after validating pre-intervention parallel trends.

## Repository Structure
- `data/raw/` & `data/processed/`: Raw 400k-row CSV and cleaned weekly/learner-level datasets.
- `src/`: Modular Python scripts for generation, cleaning, power analysis, A/B testing, and DiD.
- `notebooks/edtech_causal_analysis.ipynb`: Interactive Deliverable 2 notebook with inline visualizations.
- `reports/01_experiment_design_doc.md`: Deliverable 1 (Pre-registered A/B Test Design).
- `reports/02_decision_memo.md`: Deliverable 3 (One-Page Executive Go/No-Go Decision Memo).
- `reports/figures/`: Generated plots (`ab_test_subgroup_cuped_cis.png`, `did_parallel_trends.png`).

## Quick Start
```bash
pip install -r requirements.txt
python src/data_generator.py
python src/data_cleaner.py
python src/power_analysis.py
python src/ab_test_analyzer.py
python src/did_analyzer.py