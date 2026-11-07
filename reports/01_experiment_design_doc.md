# Experiment Design Document: AI Doubt-Solving & Adaptive Practice Initiative

## 1. Executive Context & Business Problem
Following an unvalidated product change in Ahmedabad during Q2 (`Q2_Launch`), management observed a raw increase in assessment scores. However, without a randomized control group, the institute could not separate true product impact from exam-season surges. Before rolling out our next major pedagogical initiative in Q3 (`Q3_AB_Test`)—an **Adaptive Micro-Quiz & Instant Doubt-Resolution Engine**—this document establishes a rigorous, pre-registered A/B testing protocol.

## 2. Experimental Hypothesis
* **Null Hypothesis ($H_0$):** Introducing the Adaptive Practice Engine in Ahmedabad batches has no statistically significant effect on active learner assessment scores compared to the standard curriculum ($\mu_{\text{Treatment}} - \mu_{\text{Control}} = 0$).
* **Alternative Hypothesis ($H_1$):** Introducing the Adaptive Practice Engine increases the mean `active_assessment_score` by at least **+1.50 points** ($\mu_{\text{Treatment}} - \mu_{\text{Control}} \ge 1.50$, two-sided $\alpha = 0.05$) without increasing the learner drop-off rate.

## 3. Metric Hierarchy & Definitions
To prevent metric dilution and account for historical logging flaws (e.g., ghost active drop-outs and unlogged recorded sessions), all metrics are computed at the deduplicated **unique learner grain (`learner_id`)** across the 8-week quarter:

* **Primary Evaluation Metric (Go/No-Go Driver):**
  * **Active Assessment Score (`active_assessment_score`):** Mean weekly assessment score (0–100 scale) across active weeks prior to any drop-out. Chosen over binary completion rate for superior statistical sensitivity and lower sample size requirements.
* **Secondary Metric (Total Pedagogical + Retention Yield):**
  * **Overall Assessment Score (`overall_assessment_score`):** Mean weekly score across all 8 weeks (assigning 0 to post-dropout weeks).
* **Guardrail Metrics (Must Not Degrade):**
  * **Learner Drop-off Rate (`dropped_out`):** Proportion of enrolled learners with $\ge 2$ consecutive trailing weeks of zero attendance, zero assignment submission, and zero assessment score through Week 8. Must not increase significantly ($\Delta \le 0$).
  * **Course Completion Rate (`completed_course`):** Proportion of learners who remain active through Week 8 and maintain $\ge 50\%$ imputed attendance.
  * **Missing Attendance Telemetry Rate (`missing_attendance_rate`):** Proportion of `Recorded` session logs missing raw attendance timestamps (monitors player instrumentation health).

## 4. Randomization Unit & Spillover Mitigation
* **Unit of Randomization:** Individual Learner (`learner_id`) stratified within `region == 'Ahmedabad'` during `Q3_AB_Test`, allocated 50% to `Control` and 50% to `Treatment`.
* **Spillover (SUTVA) Consideration:** Because the intervention is delivered digitally inside each learner's personalized portal (individual adaptive quizzes), learner-level randomization maximizes statistical power. Peer-to-peer classroom contamination is monitored as a potential threat to validity.

## 5. Power Analysis, Sample Size & Test Duration
Baseline parameters were estimated from cleaned historical Ahmedabad data in `Q1_Pre` ($N = 8,660$ learners, enrollment velocity $\approx 1,082$ learners/week):
* **Baseline Active Assessment Score:** Mean = $61.50$, Standard Deviation ($\sigma$) = $9.23$
* **Baseline Course Completion Rate:** $81.9\%$ (Drop-off Rate = $18.1\%$)
* **Statistical Parameters:** Two-sided significance level $\alpha = 0.05$, Statistical Power $(1 - \beta) = 0.80$, Allocation Ratio = $1:1$.

| Metric Type | Metric Name | Target MDE | Standardized Effect | Required $N$ per Arm | Total Sample Size ($N$) | Recruitment Window |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary (Unadjusted)** | `active_assessment_score` | +1.50 pts | $d = 0.1625$ | 595 | 1,190 learners | ~1.10 weeks |
| **Primary (With CUPED)** | `active_assessment_score` | +1.50 pts | $r = 0.945$ ($R^2 = 0.89$) | 64 | 128 learners | ~0.12 weeks |
| **Guardrail (Binary)** | `completed_course` | +3.0% pp | $h = 0.0806$ | 2,416 | 4,832 learners | ~4.46 weeks |

### Operational Test Duration Decision
1. **Enrollment Window:** While detecting a +1.50 pt score lift requires only **1,190 learners** (~1.1 weeks of intake), protecting our binary guardrail metric (**Course Completion Rate** at $\pm 3.0\%$ pp precision) requires **4,832 learners** (~4.5 weeks of enrollment). Evaluating the full `Q3_AB_Test` Ahmedabad cohort (~10,100 learners across the quarter) provides $>99\%$ power for the primary metric and $>95\%$ power for subgroup segmentation (`Beginner`, `Intermediate`, `Advanced`).
2. **Observation Window:** Must run for one full **8-week instructional cycle** so late-quarter drop-outs (Weeks 5–8) are fully observed before calling the test.

## 6. Pre-Registered Decision Criteria
* **GO:** `active_assessment_score` lift is statistically significant ($p < 0.05$), the 95% confidence interval lower bound exceeds $+1.00$ pt, and `dropped_out` does not worsen.
* **NO-GO:** Primary metric shows null/negative lift, or guardrail drop-off increases significantly.
* **EXTEND / ITERATE:** Overall lift is borderline ($+0.5$ to $+1.5$ pts) or concentrated exclusively in one segment while harming another.