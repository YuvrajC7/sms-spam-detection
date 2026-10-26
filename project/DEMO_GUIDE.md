# Classroom demonstration and viva preparation

**Yuvraj Chandak · 25BCE2561 · BAMAT207**

## Before class

1. Double-click `START_DEMO.cmd`; test the prize and everyday examples.
2. Keep the report PDF and executed notebook open alongside the demo.
3. Verify the GitHub URL opens and contains the submitted files.
4. Keep the submission ZIP as a backup. Initial package installation needs internet; the prepared demo does not.

## A six-minute demonstration

**0:00–0:40 — The question.** “My project predicts whether an SMS is spam or ham and shows probabilities for both. It also connects the prediction process to the probability concepts from our course.”

**0:40–1:20 — Data and cleaning.** “The official UCI source has 5,574 messages. I check missing values and labels, trim surrounding whitespace and remove 414 duplicate texts before splitting. The final dataset has 5,160 messages: 642 spam and 4,518 ham.” Show the class distribution and explain the imbalance.

**1:20–2:10 — Bayes' theorem.** “199 messages contain the whole word free, and 144 of these are spam. P(Spam given Free) is 144/199 = 0.723618. Bayes gives (144/642) × (642/5160) ÷ (199/5160), the same answer.” Point out that P(Free given Spam) is a different quantity.

**2:10–2:40 — Random variable.** “Y is one for spam and zero for ham. It is Bernoulli with p=642/5160. E(Y)=p=0.124419 and Var(Y)=p(1−p)=0.108939.”

**2:40–3:30 — Model.** “I use a stratified 80/20 split with seed 42. CountVectorizer learns word-count features only from the training data. Multinomial Naive Bayes combines the prior with token likelihoods and uses Laplace smoothing.”

**3:30–4:20 — Evaluation.** Show the confusion matrix. “Accuracy is 98.26%, compared with an always-ham baseline of 87.60%. The model finds 116 of 128 spam messages, misses 12 and falsely flags 6 genuine messages. Spam precision is 95.08% and recall is 90.625%.”

**4:20–5:30 — Live input.** Predict the prize example, an ordinary message and the ambiguous “Are you free for lunch tomorrow?”. Explain why a word such as free does not force a spam prediction. Enter the teacher's own message. If a result is wrong, acknowledge it and inspect word contributions; do not claim perfection.

**5:30–6:00 — Limits and portfolio.** Try the unknown-vocabulary example and explain the prior-only warning. “The probabilities are estimates, not calibrated certainty. Historical English SMS data does not cover all modern scams.” Show the repository and conclude with the connection between Bayes' theorem and classification.

## Likely viva questions

**Why Naive Bayes?** Word counts are nonnegative discrete features; MultinomialNB is a simple, efficient probabilistic classifier required by the assignment.

**What is naive?** It treats token evidence as conditionally independent given the class, although real words are related.

**Why remove duplicates before splitting?** An identical message in training and test would leak information and inflate apparent generalization.

**Why stratify?** To preserve approximately the spam-to-ham ratio in each partition.

**Why not train CountVectorizer on everything?** That would expose test vocabulary during fitting. Fit only on training messages, transform test and new messages afterward.

**What is alpha=1?** Laplace smoothing adds one to token counts so a term absent in one class does not have a zero estimated likelihood for that class.

**Are the five word probabilities the model features?** No. They are descriptive message-presence events. The model uses all 7,673 training vocabulary terms with frequency counts.

**Why is P(Spam given prize)=1?** All 70 observed messages containing that word are spam. This is a finite-sample proportion, not a universal rule.

**Why is the expected label fractional?** Expectation is the average of an indicator, not a possible label of one message.

**Why is accuracy insufficient?** Always predicting ham already yields 87.60%. Spam recall exposes that such a baseline finds no spam.

**What is a false positive?** A genuine message predicted as spam. A false negative is spam predicted as genuine.

**Do the model probabilities mean real-world certainty?** No. They are normalized model scores under its assumptions. Calibration would require separate validation.

**Why don't you show exactly 0.96 and 0.04?** Those were illustrative assignment values. My display uses the actual trained model probabilities.

**What happens for unknown words?** Known terms contribute evidence; unseen terms are ignored. With no known terms the result falls back to training priors and shows a warning.

**Can it handle Hindi or new scams?** Performance on those distributions has not been established by this English historical dataset.

**How would you improve it?** Compare character n-grams or bigrams and tune smoothing using training-only cross-validation. Evaluate improvements on a fresh held-out set.

## Backup terminal commands

```sh
python spam_detector.py predict "Are you free for lunch tomorrow?"
python spam_detector.py interactive
python -m unittest discover -s tests -v
```

The exact figures are saved in `results/metrics.json` and `results/word_probabilities.csv`. Never memorize a claim of 100% model accuracy: the measured errors are part of the project explanation.
