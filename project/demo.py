"""Offline classroom interface. Launch with python demo.py."""
import json
import tkinter as tk
from tkinter import ttk, messagebox
from spam_detector import ROOT, predict
import joblib

EXAMPLES = {
    'Prize message': 'Congratulations! You have won a free prize. Claim now.',
    'Everyday message': 'Can we meet at the library after class?',
    'Ambiguous word': 'Are you free for lunch tomorrow?',
    'Unknown vocabulary': 'zzzxqv qqqzzz',
}

class Demo(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('SMS Probability Lab | Yuvraj Chandak | BAMAT207')
        self.geometry('1050x780'); self.minsize(900, 700)
        self.configure(bg='#f3f6fa')
        self.model = joblib.load(ROOT / 'models/spam_pipeline.joblib')
        self.metrics = json.loads((ROOT / 'results/metrics.json').read_text())
        style = ttk.Style(self); style.theme_use('clam')
        style.configure('TFrame', background='#f3f6fa')
        style.configure('TLabel', background='#f3f6fa', foreground='#18324d', font=('Segoe UI', 11))
        style.configure('Title.TLabel', font=('Segoe UI', 25, 'bold'))
        style.configure('TButton', font=('Segoe UI', 11), padding=8)
        style.configure('TNotebook.Tab', padding=(20, 10), font=('Segoe UI', 11))
        header = ttk.Frame(self, padding=24); header.pack(fill='x')
        ttk.Label(header, text='SMS Probability Lab', style='Title.TLabel').pack(anchor='w')
        ttk.Label(header, text='From Bayes theorem to a live spam prediction  |  Yuvraj Chandak · 25BCE2561').pack(anchor='w', pady=(7,0))
        tabs = ttk.Notebook(self); tabs.pack(fill='both', expand=True, padx=24, pady=(0,20))
        live = ttk.Frame(tabs, padding=20); tabs.add(live, text='Live prediction')
        stats = ttk.Frame(tabs, padding=20); tabs.add(stats, text='Probability and evaluation')
        ttk.Label(live, text='Enter an SMS message').pack(anchor='w')
        self.input = tk.Text(live, height=4, wrap='word', font=('Segoe UI', 13), relief='solid', bd=1, padx=12, pady=12)
        self.input.pack(fill='x', pady=8)
        buttons = ttk.Frame(live); buttons.pack(fill='x')
        for label, text in EXAMPLES.items():
            ttk.Button(buttons, text=label, command=lambda t=text: self.set_example(t)).pack(side='left', padx=(0,7))
        ttk.Button(live, text='Analyse message  →', command=self.analyse).pack(anchor='w', pady=12)
        self.result = ttk.Label(live, text='Ready for your first message', font=('Segoe UI', 19, 'bold'))
        self.result.pack(anchor='w')
        self.prob = ttk.Label(live, text=''); self.prob.pack(anchor='w', pady=(6,4))
        self.bar = ttk.Progressbar(live, maximum=100); self.bar.pack(fill='x', pady=5)
        self.note = ttk.Label(live, text='Probabilities are model estimates, not guarantees.', wraplength=900)
        self.note.pack(anchor='w', pady=5)
        ttk.Label(live, text='Recognized word contributions to spam log odds').pack(anchor='w', pady=(8,5))
        self.tree = ttk.Treeview(live, columns=('word','count','effect'), show='headings', height=6)
        for col, title in [('word','Word'),('count','Count'),('effect','Contribution (+ spam / − ham)')]:
            self.tree.heading(col, text=title); self.tree.column(col, width=230, anchor='center')
        self.tree.pack(fill='both', expand=True)
        self.input.bind('<Control-Return>', lambda event: self.analyse())
        m = self.metrics; cm=m['confusion_matrix']; r=m['classification_report']['Spam']
        text = (f"DATASET\n{m['audit']['raw_rows']:,} raw rows → {m['audit']['cleaned_rows']:,} unique messages\n"
                f"Ham: {m['counts']['ham']:,}   Spam: {m['counts']['spam']:,}\n\n"
                f"BERNOULLI RANDOM VARIABLE\nP(Spam) = E(Y) = {m['p_spam']:.6f}\n"
                f"P(Ham) = {m['p_ham']:.6f}\nVar(Y) = p(1-p) = {m['variance']:.6f}\n\n"
                f"HELD-OUT TEST RESULTS\nTrain: {m['train_size']:,}   Test: {m['test_size']:,}   Seed: 42\n"
                f"Accuracy: {m['accuracy']:.2%}   Always-ham baseline: {m['baseline_accuracy']:.2%}\n"
                f"Spam precision: {r['precision']:.2%}   Recall: {r['recall']:.3%}   F1: {r['f1-score']:.2%}\n"
                f"TN: {cm[0][0]}   FP: {cm[0][1]}   FN: {cm[1][0]}   TP: {cm[1][1]}\n\n"
                "WORD PROBABILITIES\n")
        import pandas as pd
        for row in pd.read_csv(ROOT/'results/word_probabilities.csv').itertuples():
            text += f"P(Spam | {row.word}) = {row.spam_containing}/{row.containing} = {row.p_spam_given_word:.4f}\n"
        text += '\nWord proportions describe the cleaned corpus. The model learns only from training messages.'
        ttk.Label(stats, text=text, font=('Segoe UI',12), justify='left').pack(anchor='nw')
        self.set_example(EXAMPLES['Prize message'])

    def set_example(self, text):
        self.input.delete('1.0','end'); self.input.insert('1.0',text)

    def analyse(self):
        try:
            result = predict(self.input.get('1.0','end').strip(), self.model)
        except ValueError as error:
            messagebox.showinfo('Check your message', str(error)); return
        self.result.configure(text=f"Prediction: {result['prediction']}")
        self.prob.configure(text=f"P(Spam | Message) = {result['p_spam']:.10f}     P(Ham | Message) = {result['p_ham']:.10f}")
        self.bar['value'] = result['p_spam']*100
        self.note.configure(text=result['warning'] or f"{result['known_tokens']} of {result['total_tokens']} token occurrences recognized. Model estimates are not calibrated certainty.")
        for item in self.tree.get_children(): self.tree.delete(item)
        for row in result['contributions'][:8]:
            self.tree.insert('', 'end', values=(row['word'],row['count'],f"{row['log_odds']:+.3f}"))

if __name__ == '__main__':
    try:
        Demo().mainloop()
    except FileNotFoundError:
        print('Missing model or results. Run: python spam_detector.py train')
        raise SystemExit(1)
