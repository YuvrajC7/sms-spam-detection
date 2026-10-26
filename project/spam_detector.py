"""Reproducible probability analysis and SMS classification. Run --help."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import platform
import re
import urllib.request
import zipfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
URL = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
WORDS = ['free', 'prize', 'win', 'urgent', 'offer']
SEED = 42

def load_data():
    """Read the official tab-delimited file without interpreting SMS quotation marks."""
    path = ROOT / 'data/SMSSpamCollection'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            path.write_bytes(archive.read('SMSSpamCollection'))
            (path.parent / 'UCI_README.txt').write_bytes(archive.read('readme'))
    rows = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        if '\t' not in line:
            raise ValueError(f'Malformed source line {line_no}')
        label, text = line.split('\t', 1)
        rows.append({'source_line': line_no, 'label': label, 'message': text})
    raw = pd.DataFrame(rows)
    missing = int(raw[['label', 'message']].isna().any(axis=1).sum())
    df = raw.dropna(subset=['label', 'message']).copy()
    df['message'] = df.message.str.strip()
    blank = int(df.message.eq('').sum())
    df = df.loc[df.message.ne('')].copy()
    if not set(df.label).issubset({'ham', 'spam'}):
        raise ValueError('Unexpected labels in dataset')
    conflicts = int((df.groupby('message').label.nunique() > 1).sum())
    if conflicts:
        raise ValueError('Conflicting labels for identical messages require review')
    duplicates = int(df.duplicated('message').sum())
    df = df.drop_duplicates('message').reset_index(drop=True)
    df['target'] = df.label.map({'ham': 0, 'spam': 1})
    df['message_id'] = [hashlib.sha256(t.encode()).hexdigest()[:16] for t in df.message]
    audit = dict(raw_rows=len(raw), missing_rows=missing, blank_rows=blank,
                 duplicate_rows=duplicates, conflicting_messages=conflicts, cleaned_rows=len(df),
                 raw_counts=raw.label.value_counts().to_dict(),
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return df, audit

def word_probabilities(df):
    """A word event counts messages, not occurrences; matches are case insensitive."""
    p_spam = float(df.target.mean())
    rows = []
    for word in WORDS:
        contains = df.message.str.contains(r'\b' + re.escape(word) + r'\b', case=False, regex=True)
        total = int(contains.sum())
        spam = int((contains & df.target.eq(1)).sum())
        p_word = total / len(df)
        p_word_spam = spam / int(df.target.sum())
        direct = spam / total if total else None
        bayes = p_word_spam * p_spam / p_word if total else None
        rows.append(dict(word=word, containing=total, spam_containing=spam,
                         ham_containing=total-spam, p_word=p_word,
                         p_word_given_spam=p_word_spam, p_spam_given_word=direct,
                         bayes_verification=bayes))
    return pd.DataFrame(rows)

def predict(message, model=None):
    if not isinstance(message, str) or not message.strip():
        raise ValueError('Please enter a non-empty SMS message.')
    if len(message) > 10000:
        raise ValueError('Please keep the message below 10,001 characters.')
    if model is None:
        path = ROOT / 'models/spam_pipeline.joblib'
        if not path.exists():
            raise FileNotFoundError('Train first: python spam_detector.py train')
        model = joblib.load(path)  # Load only this locally generated trusted artifact.
    vectorizer = model.named_steps['vectorizer']
    features = vectorizer.transform([message])
    probabilities = dict(zip(model.classes_.tolist(), model.predict_proba([message])[0].tolist()))
    tokens = vectorizer.build_analyzer()(message)
    known = [t for t in tokens if t in vectorizer.vocabulary_]
    # Exact additive log-odds contributions for recognized tokens.
    nb = model.named_steps['classifier']
    contributions = []
    for index, count in zip(features.indices, features.data):
        term = vectorizer.get_feature_names_out()[index]
        delta = float(count * (nb.feature_log_prob_[1, index] - nb.feature_log_prob_[0, index]))
        contributions.append(dict(word=term, count=int(count), log_odds=delta))
    contributions.sort(key=lambda x: abs(x['log_odds']), reverse=True)
    return dict(prediction='Spam' if model.predict([message])[0] else 'Ham',
                p_spam=probabilities[1], p_ham=probabilities[0],
                known_tokens=len(known), total_tokens=len(tokens),
                warning='No recognized vocabulary. Probabilities equal training class priors.' if not known else '',
                contributions=contributions,
                prior_log_odds=float(nb.class_log_prior_[1]-nb.class_log_prior_[0]))

def train():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for directory in ['models', 'results']:
        (ROOT / directory).mkdir(exist_ok=True)
    df, audit = load_data()
    df.to_csv(ROOT / 'data/cleaned_sms.csv', index=False)
    words = word_probabilities(df)
    words.to_csv(ROOT / 'results/word_probabilities.csv', index=False)
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2,
                                          random_state=SEED, stratify=df.target)
    tr, te = df.iloc[train_idx], df.iloc[test_idx]
    model = Pipeline([('vectorizer', CountVectorizer()), ('classifier', MultinomialNB())])
    model.fit(tr.message, tr.target)
    pred = model.predict(te.message)
    prob = model.predict_proba(te.message)[:, 1]
    cm = confusion_matrix(te.target, pred, labels=[0, 1])
    report = classification_report(te.target, pred, labels=[0, 1],
                                   target_names=['Ham', 'Spam'], output_dict=True, zero_division=0)
    report_text = classification_report(te.target, pred, labels=[0, 1],
                                        target_names=['Ham', 'Spam'], digits=6, zero_division=0)
    (ROOT / 'results/classification_report.txt').write_text(report_text, encoding='utf-8')
    predictions = te.copy()
    predictions['prediction'] = pred
    predictions['p_spam'] = prob
    predictions['error_type'] = np.where((te.target == 1) & (pred == 0), 'False negative',
                               np.where((te.target == 0) & (pred == 1), 'False positive', 'Correct'))
    predictions.to_csv(ROOT / 'results/test_predictions.csv', index=False)
    predictions[predictions.error_type != 'Correct'].to_csv(ROOT / 'results/errors.csv', index=False)
    split = df[['message_id', 'source_line', 'label']].copy()
    split['partition'] = 'train'
    split.loc[test_idx, 'partition'] = 'test'
    split.to_csv(ROOT / 'results/split_manifest.csv', index=False)
    joblib.dump(model, ROOT / 'models/spam_pipeline.joblib')
    p = float(df.target.mean())
    metrics = dict(audit=audit, counts=df.label.value_counts().to_dict(), p_spam=p, p_ham=1-p,
                   expectation=p, variance=p*(1-p), train_size=len(tr), test_size=len(te),
                   train_counts=tr.label.value_counts().to_dict(), test_counts=te.label.value_counts().to_dict(),
                   vocabulary_size=len(model.named_steps['vectorizer'].vocabulary_),
                   random_state=SEED, confusion_matrix=cm.tolist(), classification_report=report,
                   accuracy=float(accuracy_score(te.target, pred)), baseline_accuracy=float(te.target.eq(0).mean()),
                   versions=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                                 pandas=pd.__version__, numpy=np.__version__, joblib=joblib.__version__),
                   demo=predict('Congratulations! You have won a free prize. Claim now.', model))
    (ROOT / 'results/metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 150})
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(['Ham', 'Spam'], [metrics['counts']['ham'], metrics['counts']['spam']], color=['#167d9a', '#db6842'], width=.55)
    ax.bar_label(bars, padding=4)
    ax.set(ylabel='Number of unique messages', title='Class distribution after duplicate removal', ylim=(0, metrics['counts']['ham']*1.15))
    fig.tight_layout(); fig.savefig(ROOT / 'results/class_distribution.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.imshow(cm, cmap='Blues')
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f'{value}', ha='center', va='center', fontsize=19, color='white' if value > cm.max()/2 else '#182b44')
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
           xlabel='Predicted label', ylabel='Actual label', title='Held-out test confusion matrix')
    fig.tight_layout(); fig.savefig(ROOT / 'results/confusion_matrix.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(words.word, words.p_spam_given_word, color='#167d9a')
    ax.bar_label(bars, labels=[f'{v:.1%}' for v in words.p_spam_given_word], padding=4)
    ax.axhline(p, linestyle='--', color='#db6842', label=f'Prior P(Spam) = {p:.1%}')
    ax.set(ylim=(0,1.12), ylabel='Empirical conditional probability', title='How informative is each suspicious word?')
    ax.legend(loc='upper right'); fig.tight_layout()
    fig.savefig(ROOT / 'results/word_probabilities.png'); plt.close(fig)
    return metrics

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('train', help='Reproduce analysis, model, charts and evaluation')
    p = sub.add_parser('predict', help='Classify one SMS')
    p.add_argument('message')
    sub.add_parser('interactive', help='Enter messages repeatedly; empty input exits')
    args = parser.parse_args()
    if args.command == 'train':
        print(json.dumps(train(), indent=2))
    elif args.command == 'predict':
        print(json.dumps(predict(args.message), indent=2))
    else:
        while True:
            try:
                message = input('\nSMS (blank to exit): ')
                if not message.strip(): break
                result = predict(message)
                print(f"{result['prediction']} | P(Spam)={result['p_spam']:.6f} | P(Ham)={result['p_ham']:.6f}")
                if result['warning']: print(result['warning'])
            except (EOFError, KeyboardInterrupt): break
            except ValueError as error: print(error)

if __name__ == '__main__':
    main()
