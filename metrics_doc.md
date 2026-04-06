# Fairness and Capability Metrics

This document summarizes the evaluation metrics used for the datasets:

* **MMLU** (capability benchmark)
* **StereoSet** (stereotype bias benchmark)
* **BBQ** (Bias Benchmark for Question Answering)
* **WinoBias** (gender bias in coreference resolution)

Each section contains:

* Formula
* Description
* Interpretation of values
* Optimization direction

---

# 1. MMLU (Massive Multitask Language Understanding)

### Metric: Accuracy

### Formula

$$
Accuracy = \frac{\text{Number of Correct Answers}}{\text{Total Number of Questions}}
$$

### Description

MMLU evaluates **general reasoning and knowledge across multiple domains** such as:

* mathematics
* law
* medicine
* physics
* humanities

The metric measures the **fraction of correctly answered multiple-choice questions**.

### Range

```
[0,1]  (often reported as percentage [0,100])
```

### Interpretation

| Value | Meaning                                               |
| ----- | ----------------------------------------------------- |
| High  | Model has strong reasoning and knowledge capabilities |
| Low   | Model struggles with understanding tasks              |

### Optimization Direction

```
Higher is better
```

### Effect of Quantization

* Aggressive quantization (W4, W3) typically **reduces accuracy**
* Capability degradation increases as precision decreases.

---

# 2. StereoSet

StereoSet measures **stereotype bias in language models**.

It evaluates whether the model prefers **stereotypical sentences** over **anti-stereotypical ones**.

Two metrics are used.

---

# 2.1 StereoSet Metric 1 — Stereotype Score

### Formula

$$
StereotypeScore =
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
\left[
ppl(s_{more}) < ppl(s_{less})
\right]
$$

Where:

* $ s_{more} $ = stereotypical sentence
* $ s_{less} $ = anti-stereotypical sentence
* $ ppl(\cdot) $ = perplexity
* $ \mathbf{1}[\cdot] $ = indicator function

Indicator function:

```
1 if condition is true
0 otherwise
```

### Intuition

Lower perplexity means the model finds the sentence **more natural**.

Thus this metric counts how often:

```
stereotypical sentence is preferred
```

### Range

```
[0,1]   (reported as [0,100])
```

### Interpretation

| Value | Meaning                                        |
| ----- | ---------------------------------------------- |
| > 0.5(50) | Model prefers stereotypical sentences (biased) |
| = 0.5(50) | No preference (ideal)                          |
| < 0.5(50) | Prefers anti-stereotypical sentences           |

### Target Value

```
0.5(50)
```

### Important Observation

Quantization often moves the score **toward 0.5** because:

* overall likelihood decreases
* model uncertainty increases

This **does NOT necessarily mean bias is reduced**.

---

# 2.2 StereoSet Metric 2 — Effect Size (Cohen's d)

### Formula

$$
d =
\frac{\mu_{more} - \mu_{less}}
{\sigma_{pooled}}
$$

Where:

* $ \mu_{more} $ = mean log-likelihood of stereotypical sentences
* $ \mu_{less} $ = mean log-likelihood of anti-stereotypical sentences
* $ \sigma_{pooled} $ = pooled standard deviation

### Description

Measures **magnitude of preference** between stereotypical and anti-stereotypical sentences.

### Range

```
(-∞ , +∞)
```

### Interpretation

| Value    | Meaning                                    |
| -------- | ------------------------------------------ |
| Positive | Model prefers stereotypical sentences      |
| Zero     | No bias                                    |
| Negative | Model prefers anti-stereotypical sentences |

### Target Value

```
0
```

### Quantization Effect

Quantization can push the value **negative**, meaning:

```
model appears to prefer anti-stereotypes
```

But this may be due to **general uncertainty rather than real debiasing**.

---

# 3. BBQ (Bias Benchmark for QA)

BBQ evaluates bias when answering **multiple-choice questions involving social groups**.

Two settings exist:

1. **Ambiguous Context**
2. **Disambiguated Context**

---

# 3.1 BBQ Metric — Ambiguous Context Bias

### Formula

$$
s_{amb} =
\frac{n_{pro} - n_{anti}}
{n_{pro} + n_{anti} + n_{unbiased}}
$$

Where:

* $ n_{pro} $ = stereotypical answers
* $ n_{anti} $ = anti-stereotypical answers
* $ n_{unbiased} $ = neutral answers

### Description

In ambiguous context:

```
the correct answer cannot be determined
```

Any preference for a stereotype indicates bias.

### Range

```
[-1 , 1]  (often scaled to [-100 , 100])
```

### Interpretation

| Value    | Meaning                                  |
| -------- | ---------------------------------------- |
| Positive | Model prefers stereotypical answers      |
| Zero     | No bias                                  |
| Negative | Model prefers anti-stereotypical answers |

### Target

```
0
```

---

# 3.2 BBQ Metric — Disambiguated Context Bias

### Formula

$$
s_{disamb} =
\frac{n_{pro} - n_{anti}}
{n_{pro} + n_{anti}}
$$

### Description

In this setup the **correct answer is known from context**.

If the model still chooses stereotypical answers, it indicates bias.

### Range

```
[-1 , 1]
```

### Interpretation

| Value         | Meaning                                    |
| ------------- | ------------------------------------------ |
| High positive | Model ignores context and uses stereotypes |
| Zero          | Fair decision                              |
| Negative      | Prefers anti-stereotypes                   |

### Target

```
0
```

### Quantization Effect

Quantization tends to:

* **increase ambiguous bias**
* have **little effect on disambiguated bias**

---

# 4. WinoBias

WinoBias measures **gender bias in coreference resolution**.

Example:

```
The nurse helped the patient because she was kind.
```

The model must determine **who "she" refers to**.

Two metrics are used.

---

# 4.1 Historical Bias

### Formula

$$
HistoricalBias =
Acc_{pro} - Acc_{anti}
$$

Where:

* $ Acc_{pro} $ = accuracy on stereotypical sentences
* $ Acc_{anti} $ = accuracy on anti-stereotypical sentences

### Range

```
[-100 , 100]  (in percentage difference)
```

### Interpretation

| Value    | Meaning                            |
| -------- | ---------------------------------- |
| Positive | Better at stereotypical cases      |
| Zero     | No bias                            |
| Negative | Better at anti-stereotypical cases |

### Target

```
0
```

### Quantization Effect

Accuracy tends to drop more for **anti-stereotypical sentences**, increasing bias.

---

# 4.2 Population Bias

### Formula

$$
PopulationBias =
|Acc_{male} - Acc_{female}|
$$

Where:

* $ Acc_{male} $ = accuracy when pronoun is male
* $ Acc_{female} $ = accuracy when pronoun is female

### Range

```
[0 , 100]
```

### Interpretation

| Value | Meaning                |
| ----- | ---------------------- |
| High  | Large gender disparity |
| Zero  | Equal performance      |

### Target

```
0
```

### Quantization Effect

Usually **small impact (<3%)**, but can vary across models.

---

# Summary Table

| Dataset   | Metric             | Target | Direction |
| --------- | ------------------ | ------ | --------- |
| MMLU      | Accuracy           | High   | ↑         |
| StereoSet | StereotypeScore    | 0.5    | →         |
| StereoSet | Cohen's d          | 0      | →         |
| BBQ       | Ambiguous Bias     | 0      | ↓         |
| BBQ       | Disambiguated Bias | 0      | ↓         |
| WinoBias  | Historical Bias    | 0      | ↓         |
| WinoBias  | Population Bias    | 0      | ↓         |

---

# Key Insight (From the Paper)

Probability-based metrics and generation-based metrics may **contradict each other**.

Example:

* StereoSet may show **bias decreasing**
* BBQ / WinoBias may show **bias increasing**

Reason:

```
Quantization increases model uncertainty
```

This reduces log-likelihood differences, giving the illusion of fairness.

Therefore **multiple benchmarks are necessary** when studying fairness under quantization.

---
