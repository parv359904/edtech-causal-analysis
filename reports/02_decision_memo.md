# Executive Decision Memo: Causal Impact of Q2 & Q3 Pedagogical Initiatives

**To:** Academic Leadership & Product Engineering, Ahmedabad Coaching Institute Chain  
**From:** Causal Inference & Experimentation Team  
**Subject:** Post-Mortem on Q2 Unvalidated Launch (DiD) & Go/No-Go Decision on Q3 Adaptive Practice A/B Test  

---

## 1. Executive Summary & Final Recommendation
* **Q3 Adaptive Practice Initiative (A/B Test): GO (Immediate Full Rollout).** The randomized controlled trial ($N = 10,106$ learners in Ahmedabad) demonstrates a statistically significant increase in **Active Assessment Score of +3.43 points** (CUPED 95% CI: **[+3.31, +3.55]**, $p < 0.0001$) alongside a **-3.92% percentage-point reduction in learner drop-off** (95% CI: **[-5.34%, -2.50%]**, $p < 0.0001$).
* **Q2 Unvalidated Curriculum Change (Observational DiD): PARTIAL / CAUTIONARY.** While management observed a naive **+4.71 point** score jump in Ahmedabad during Q2, Difference-in-Differences (DiD) against Vadodara reveals that **+3.50 points (~74% of the gain)** was driven by exam-seasonality rather than the product change. The true causal effect of the Q2 change was a modest **+1.21 points** (Cluster-Robust 95% CI: **[+1.06, +1.36]**, $p < 0.0001$), with negligible impact on student retention.

---

## 2. Data Integrity Audit (400,000 Raw Records)
Prior to modeling, three instrumentation flaws in the raw telemetry (`400,000` rows) were remediated:
1. **Duplicate Batch Enrolments:** Removed `15,000` duplicate `(learner_id, batch_id, week)` records (`3.75%` inflation) that would otherwise artificially shrink standard errors.
2. **Ghost Active Drop-outs:** Reclassified `17.3%` of learners who exhibited uninterrupted zero attendance, assignments, and scores through Week 8 from `'Active'` to `'Dropped_Out'`, preventing retention shifts from confounding active academic performance.
3. **Missing Recorded Session Attendance:** Imputed `94,000+` missing (`NaN`) attendance logs (`70%` of `Recorded` mode rows) using verified weekly assignment and assessment submissions, recovering true hybrid attendance (`~85%`).

---

## 3. Summary of Causal Estimates & Confidence Intervals
*Note: Point estimates and confidence intervals reflect exact outputs from the cleaned synthetic cohort ($N = 48,125$ learners).*

| Evaluation Framework | Metric / Estimand | Point Estimate | 95% Confidence Interval | $p$-value | Assumption & Sanity Check Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q3 A/B Test (Unadjusted Welch)** | Primary: Active Score (pts) | +3.43 pts | [+3.07, +3.79] | $< 10^{-16}$ | SRM $\chi^2 p = 0.72$ (Pass); $\|\text{SMD}\| < 0.02$ |
| **Q3 A/B Test (Bootstrap 2,000x)** | Primary: Active Score (pts) | +3.43 pts | [+3.08, +3.79] | $< 10^{-16}$ | Non-parametric distribution check (Pass) |
| **Q3 A/B Test (CUPED Adjusted)** | Primary: Active Score (pts) | +3.43 pts | [+3.31, +3.55] | $< 10^{-16}$ | Variance reduced by **88.6%** ($r = 0.94$) |
| **Q3 A/B Test (Guardrail)** | Learner Drop-off Rate | -3.92% pp | [-5.34%, -2.50%] | $< 10^{-7}$ | Retention improved significantly |
| **Q3 A/B Test (Guardrail)** | Course Completion Rate | +3.94% pp | [+2.52%, +5.36%] | $< 10^{-7}$ | Exceeds +3.0% guardrail threshold |
| **Q2 Launch (DiD vs. Vadodara)** | Active Score (Cov-Adjusted) | +1.21 pts | [+1.06, +1.36] | $< 10^{-16}$ | Pre-trend slope diff $= +0.002$/wk ($p = 0.89$, Pass) |
| **Q2 Launch (DiD Guardrail)** | Learner Drop-off Rate | -0.15% pp | [-1.42%, +1.12%] | $0.814$ | Null retention effect |

---

## 4. Subgroup Segmentation (Who Gained or Lost?)
* **Academic Tier Heterogeneity (Q3 A/B Test):**
  * **Beginner Tier ($N = 3,521$):** Largest beneficiary with a **+4.61 point** lift (CUPED 95% CI: **[+4.41, +4.81]**). Instant doubt resolution likely prevents early conceptual bottlenecks.
  * **Intermediate ($N = 4,562$) & Advanced ($N = 2,023$):** Consistent gains of **+2.81 pts** (95% CI: **[+2.64, +2.98]**) and **+2.77 pts** (95% CI: **[+2.50, +3.04]**) respectively. No segment experienced negative impact.
* **Delivery Mode Parity:** Comparable gains across **Live** (+3.44 pts) and **Recorded** (+3.41 pts) learners once missing attendance logs were properly imputed.
* **Efficiency Gain via CUPED:** Conditioning on pre-enrolment `baseline_score` reduced variance by **88.6%**, showing that future A/B tests can reach 80% power for a +1.50 pt MDE with **~128 total learners (~0.12 weeks of intake)** instead of **1,190 learners (~1.10 weeks)**.

---

## 5. Threats to Causal Validity (What Could Make Our Estimates Wrong?)
1. **SUTVA / Classroom Spillover (A/B Test):** Because randomization occurred at the `learner_id` level within shared batches (`batch_id`), Treatment students might share AI-generated quiz explanations with Control peers in study groups. *Direction of Bias:* Typically attenuates (understates) the true treatment effect. Future tests involving collaborative features should cluster-randomize by `batch_id`.
2. **Differential Attrition / Survivorship Bias:** Because Treatment reduced drop-offs by 3.92% pp, weaker learners who would have dropped out in Control remained active in Treatment. While CUPED adjustment on `baseline_score` mitigates observable selection, unmeasured motivation differences among retained learners could slightly bias `active_assessment_score`.
3. **Time-Varying Regional Shocks in DiD:** Even though `Q1_Pre` parallel trends held ($p = 0.89$), DiD assumes no unobserved shock hit Ahmedabad exclusively in `Q2_Launch` (e.g., local competitor closure, star faculty hires, or regional schedule shifts). If such a shock occurred, the +1.21 pt DiD estimate would conflate that shock with the Q2 product change.
4. **Imputation Misclassification in Recorded Sessions:** Inferring attendance from assignment/quiz submissions assumes learners who watched recordings without submitting assignments were absent, potentially undercounting passive video consumption.