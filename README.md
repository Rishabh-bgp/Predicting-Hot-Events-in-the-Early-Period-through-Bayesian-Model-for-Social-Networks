# Early-Stage Hot Event Prediction in Social Networks

**A Bayesian Modeling Framework using Semi-Naive Bayes (BEEP)**

[![Paper](https://img.shields.io/badge/Paper-IJITCE%202025-blue)](https://ijitce.org/index.php/ijitce/article/view/1532)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> Predicting which social-media events will go viral — in the first hour — when data is sparse, noisy, and incomplete.

This repository accompanies the paper:

> **Er. Rishabh Aryan & Dr. Bhanu Priya.**  
> *Early-Stage Hot Event Prediction In Social Networks Using A Bayesian Modeling Framework.*  
> *International Journal of Information Technology & Computer Engineering*, Vol. 13, Issue 4, 2025, pp. 395–402.  
> [https://ijitce.org/index.php/ijitce/article/view/1532](https://ijitce.org/index.php/ijitce/article/view/1532)

---

## Why this exists

Traditional cascade / virality models need days of history and heavy feature engineering. By then the window to act is gone.

This work targets the **first 15–60 minutes** after an event appears. It uses a **Semi-Naive Bayes** classifier (BEEP-style) that:

- models five early-stage features with explicit distributions
- allows a controlled dependency (Acceleration | Velocity) instead of full independence
- updates posteriors with learned class priors
- stays cheap enough for real-time scoring

On Twitter + Weibo evaluation in the paper, the framework reports **87.3% accuracy** within the first hour (precision 84.6%, recall 89.2%, F1 86.8%), outperforming standard Naive Bayes, logistic regression, random forest, and SVM on the same early-window setting.

---

## Features used

| Feature | Role | Assumed distribution |
|---|---|---|
| **Sum** | Cumulative activity in the observation window | Gamma |
| **Velocity** | Sharing rate (posts / unit time) | Gamma |
| **Acceleration** | Change in velocity | Normal, **conditioned on Velocity bins** |
| **Tweet Chain** | Cascade / reply-retweet chain length | Gamma |
| **Community** | Early-adopter community / clustering signal | Gamma |

Acceleration is not treated as independent of Velocity. During `fit`, Velocity is binned and a Normal MLE is estimated **per bin** so \(P(A \mid V, H)\) is piecewise.

---

## Model in one paragraph

Let \(H \in \{H_0, H_1\}\) be non-hot vs hot.

\[
P(H \mid \mathbf{x}) \propto P(H)\, P(S \mid H)\, P(V \mid H)\, P(A \mid V, H)\, P(C_{\text{chain}} \mid H)\, P(C_{\text{comm}} \mid H)
\]

- Class priors \(P(H)\) are either uniform or **learned from training frequencies**.
- Gamma / Normal parameters are **maximum-likelihood estimates** per class.
- `predict_proba` multiplies the factorized likelihoods and normalizes over \(\{H_0, H_1\}\).

Learned priors often leave **AUC almost unchanged** (ranking is similar) while shifting **calibration** of the raw probabilities — which is why the notebook reports both AUC and Brier score.

---

## Repository layout

```
.
├── Predicting_Hot_Events_in_the_Early_Period_through_Bayesian_Model_for_Social_Networks.ipynb
├── README.md
└── (optional) src/beep.py   # extract the classifier if you split the notebook
```

The notebook is self-contained:

1. Feature extractors  
2. `BEEPClassifier` (`fit` + `predict_proba`)  
3. Mock datasets (n = 100 and n = 1000)  
4. AUC + Brier evaluation, with / without learned priors  
5. Discussion of calibration vs discrimination  

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn scipy matplotlib
jupyter notebook Predicting_Hot_Events_in_the_Early_Period_through_Bayesian_Model_for_Social_Networks.ipynb
```

Minimal usage once the class is in scope:

```python
from beep import BEEPClassifier   # or run the notebook cells

clf = BEEPClassifier()
clf.fit(X_train, y_train)         # y in {"H0", "H1"}
proba = clf.predict_proba(X_test) # columns: P(H0), P(H1)
```

`X` columns expected: `Sum`, `Velocity`, `Acceleration`, `Tweet Chain`, `Community`.

---

## Results snapshot

**Paper (Twitter + Weibo, ~101k events, 60-minute window)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **Semi-Naive Bayes (proposed)** | **87.3** | **84.6** | **89.2** | **86.8** |
| Random Forest | 82.4 | 78.3 | 84.5 | 81.3 |
| Logistic Regression | 80.2 | 75.8 | 82.7 | 79.1 |
| SVM | 79.8 | 74.1 | 83.8 | 78.6 |
| Standard Naive Bayes | 78.5 | 72.4 | 81.3 | 76.6 |

Feature contribution (ablation in the paper): Temporal velocity **34.2%**, user influence **27.8%**, network structure **19.5%**, content sentiment **11.3%**, topic relevance **7.2%**.

**Notebook (synthetic data, for unit-testing the estimator)**

| Setting | AUC | Brier |
|---|---:|---:|
| Small set (n=100), ± learned priors | ~0.96–0.99 | ~0.05–0.07 |
| Large set (n=1000) | ~0.93–0.95 | ~0.10 |

Numbers in the notebook vary slightly across cells because mock data is regenerated; they are **not** the paper’s Twitter/Weibo numbers. Use them to check that `fit` / `predict_proba` run and that priors affect calibration more than ranking.

---

## How the classifier is trained

`fit(X, y)`:

1. Split rows by label \(H_0\) / \(H_1\).
2. Learn \(P(H)\).
3. Fit Gamma MLE on Sum, Velocity, Tweet Chain, Community (per class).
4. Bin Velocity; fit Normal \((\mu, \sigma)\) on Acceleration inside each bin (per class).
5. Store parameters for scoring.

`predict_proba(X)`:

1. Evaluate each factor’s density at the observed feature.
2. Multiply by the prior.
3. Normalize the two class scores so they sum to 1.

---

## Datasets (paper)

| Platform | Events | Hot rule (paper) |
|---|---:|---|
| Twitter | 52,847 | ≥ 10,000 shares in 24 h |
| Weibo | 48,293 | ≥ 50,000 shares in 24 h |

Early window: **first hour** after detection (typically 5–10% of eventual engagement).  
Train / val / test: 70% / 15% / 15%, stratified.

This public repo ships the **method + mock-data notebook**. Raw Twitter / Weibo dumps are not redistributed here (platform ToS). Point `fit` at your own early-window feature table if you have access.

---

## Citation

```bibtex
@article{aryan2025early,
  title   = {Early-Stage Hot Event Prediction In Social Networks Using A Bayesian Modeling Framework},
  author  = {Aryan, Rishabh and Priya, Bhanu},
  journal = {International Journal of Information Technology and Computer Engineering},
  volume  = {13},
  number  = {4},
  pages   = {395--402},
  year    = {2025},
  url     = {https://ijitce.org/index.php/ijitce/article/view/1532}
}
```

Related method this implementation follows closely:

```bibtex
@inproceedings{ma2017beep,
  title     = {BEEP: A Bayesian Perspective Early Stage Event Prediction Model for Online Social Networks},
  author    = {Ma, Xin and Gao, Xiaofeng and Chen, Guihai},
  booktitle = {2017 IEEE International Conference on Data Mining (ICDM)},
  pages     = {327--336},
  year      = {2017},
  publisher = {IEEE}
}
```

---

## Authors

- **Er. Rishabh Aryan** — M.Tech (AI & Data Science), CSE, IIIT Bhagalpur  
  `rishabh_250201011@iiitbh.ac.in`
- **Dr. Bhanu Priya** — Assistant Professor (Temporary), ECE, IIIT Bhagalpur  
  `bpriya.ece@iiitbh.ac.in`

---

## Limitations (honest)

- Binary hot / not-hot is a coarse cut of a continuous popularity spectrum.
- Mock notebook data is for plumbing, not a substitute for the paper’s platforms.
- Acceleration–Velocity coupling is a **binned Normal**, not a full joint density.
- Cross-platform transfer (TikTok, Instagram, etc.) is not claimed.

---

## License

MIT — see `LICENSE` if present. Paper text remains under the journal’s copyright; cite the article if you use the results.
