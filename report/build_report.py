"""Create the report from the project's measured results, never hard-coded scores."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent/'project'
m=json.loads((PROJECT/'results/metrics.json').read_text())
w=pd.read_csv(PROJECT/'results/word_probabilities.csv')
FIGURES=HERE/'figures'
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':150})
f,ax=plt.subplots(figsize=(7,4))
bars=ax.bar(['Ham','Spam'],[m['counts']['ham'],m['counts']['spam']],color=['#167d9a','#db6842'],edgecolor='black',width=.55)
ax.bar_label(bars,padding=4)
ax.set(ylabel='Number of unique messages',title='Class distribution after duplicate removal',ylim=(0,m['counts']['ham']*1.15))
f.tight_layout();f.savefig(FIGURES/'class_distribution.png');plt.close(f)
cm=np.array(m['confusion_matrix'])
f,ax=plt.subplots(figsize=(6,4.2))
ax.imshow(cm,cmap='Blues')
for (i,j),value in np.ndenumerate(cm):
    ax.text(j,i,str(value),ha='center',va='center',fontsize=19,color='white' if value>cm.max()/2 else 'black')
ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Ham','Spam'],yticklabels=['Ham','Spam'],xlabel='Predicted label',ylabel='Actual label',title='Held-out test confusion matrix')
f.tight_layout();f.savefig(FIGURES/'confusion_matrix.png');plt.close(f)
f,ax=plt.subplots(figsize=(7,4))
bars=ax.bar(w.word,w.p_spam_given_word,color='#167d9a',edgecolor='black')
ax.bar_label(bars,labels=[f'{v:.1%}' for v in w.p_spam_given_word],padding=4)
ax.axhline(m['p_spam'],linestyle='--',color='#db6842',label=f"Prior P(Spam) = {m['p_spam']:.1%}")
ax.set(ylim=(0,1.12),ylabel='Empirical conditional probability',title='How informative is each suspicious word?')
ax.legend(loc='upper right');f.tight_layout();f.savefig(FIGURES/'word_probabilities.png');plt.close(f)
doc=Document()
sec=doc.sections[0]
sec.page_width=Inches(8.27); sec.page_height=Inches(11.69)
sec.top_margin=Inches(.72); sec.bottom_margin=Inches(.65)
sec.left_margin=sec.right_margin=Inches(.8)
sec.footer_distance=Inches(.3)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3']:
    s=doc.styles[name]; s.font.name='Calibri'; s.font.color.rgb=RGBColor(0,0,0)
for element in doc.styles.element.xpath('.//w:pBdr'):
    element.getparent().remove(element)
doc.styles['Normal'].font.size=Pt(11)
doc.styles['Normal'].paragraph_format.space_after=Pt(6)
doc.styles['Normal'].paragraph_format.line_spacing=1.05
doc.styles['Title'].font.size=Pt(29)
doc.styles['Heading 1'].font.size=Pt(20)
doc.styles['Heading 1'].paragraph_format.space_after=Pt(12)
doc.styles['Heading 2'].font.size=Pt(13)
doc.styles['Heading 2'].paragraph_format.space_before=Pt(10)
doc.styles['Heading 2'].paragraph_format.space_after=Pt(6)
doc.core_properties.title='Probabilistic SMS Spam Detection'
doc.core_properties.author='Yuvraj Chandak'
doc.core_properties.subject='BAMAT207 Probability and Statistics project report'
footer=sec.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
run=footer.add_run('BAMAT207  |  Yuvraj Chandak  |  '); run.font.size=Pt(9)
field=OxmlElement('w:fldSimple'); field.set(qn('w:instr'),'PAGE'); footer._p.append(field)

def p(text,style=None): return doc.add_paragraph(text,style)
def h(text): return doc.add_heading(text,2)
def page(title):
    heading=doc.add_heading(title,1)
    heading.paragraph_format.page_break_before=True
def formula(text):
    para=p(text); para.paragraph_format.left_indent=Inches(.18)
    for r in para.runs: r.bold=True
def table(headers,rows,widths=None):
    t=doc.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER
    t.autofit=False
    for i,x in enumerate(headers): t.rows[0].cells[i].text=str(x)
    for row in rows:
        for c,x in zip(t.add_row().cells,row): c.text=str(x)
    if widths:
        for row in t.rows:
            for c,width in zip(row.cells,widths): c.width=Inches(width)
        for c,width in zip(t.columns,widths): c.width=Inches(width)
    for ri,row in enumerate(t.rows):
        trpr=row._tr.get_or_add_trPr()
        no=OxmlElement('w:cantSplit'); trpr.append(no)
        if ri==0:
            repeat=OxmlElement('w:tblHeader'); trpr.append(repeat)
        for ci,c in enumerate(row.cells):
            c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=c._tc.get_or_add_tcPr()
            borders=OxmlElement('w:tcBorders')
            for edge in ['top','left','bottom','right']:
                e=OxmlElement('w:'+edge); e.set(qn('w:val'),'single'); e.set(qn('w:sz'),'6'); e.set(qn('w:color'),'000000'); borders.append(e)
            pr.append(borders)
            shade=OxmlElement('w:shd'); shade.set(qn('w:val'),'clear'); shade.set(qn('w:fill'),'FFFFFF'); pr.append(shade)
            margins=OxmlElement('w:tcMar')
            for edge in ['top','bottom','left','right']:
                e=OxmlElement('w:'+edge); e.set(qn('w:w'),'85'); e.set(qn('w:type'),'dxa'); margins.append(e)
            pr.append(margins)
            for para in c.paragraphs:
                para.paragraph_format.space_after=Pt(2); para.paragraph_format.space_before=Pt(2)
                para.paragraph_format.line_spacing=1
                for r in para.runs:
                    r.font.size=Pt(9.5); r.bold=(ri==0)
                    r.font.color.rgb=RGBColor(0,0,0)
    spacer=p(''); spacer.paragraph_format.space_after=Pt(0)
    spacer.paragraph_format.line_spacing=Pt(3)
    spacer.paragraph_format.space_before=Pt(0)
    return t
def fig(name,caption,width=6.2):
    para=doc.add_paragraph(); para.alignment=WD_ALIGN_PARAGRAPH.CENTER
    para.add_run().add_picture(str(FIGURES/name),width=Inches(width))
    para.paragraph_format.space_after=Pt(3)
    para.paragraph_format.keep_with_next=True
    c=p(caption); c.paragraph_format.space_after=Pt(8)
    for r in c.runs: r.font.size=Pt(9); r.italic=True

p('Probabilistic SMS Spam Detection','Title')
p('Project report for Probability and Statistics','Subtitle')
p('BAMAT207  |  Probability and Random Variables')
p('Submitted by\nYUVRAJ CHANDAK\nRegistration number 25BCE2561')
p('Submitted to\nDR. VEERAMANI S')
p('VELLORE INSTITUTE OF TECHNOLOGY\nVIT Vellore')
p('Submission date 27 October 2026\nAnalysis completed 26 October 2026')
h('Abstract')
p(f'This project implements a Python system that classifies an SMS as spam or genuine text and displays estimated probabilities for both classes. It connects empirical probability, Bayes’ theorem and a Bernoulli random variable with a Multinomial Naive Bayes classifier. The official UCI source contains {m["audit"]["raw_rows"]:,} records. After checking missing values and removing {m["audit"]["duplicate_rows"]} duplicate messages following whitespace trimming, {m["audit"]["cleaned_rows"]:,} messages remain. Five suspicious words are analysed using message-level conditional probabilities, and each result is verified through Bayes’ theorem.')
p(f'A stratified 80/20 split produces {m["train_size"]:,} training messages and {m["test_size"]:,} test messages. A CountVectorizer and MultinomialNB pipeline achieves {m["accuracy"]:.2%} test accuracy, {m["classification_report"]["Spam"]["precision"]:.2%} spam precision and {m["classification_report"]["Spam"]["recall"]:.3%} spam recall. The model misses 12 spam messages and incorrectly flags 6 genuine messages. A desktop interface, command-line predictor and executed notebook support a live classroom demonstration. Results are reproducible using the saved dataset, split manifest and model. The probabilities are model estimates and should not be interpreted as calibrated certainty.')
p('Project repository\nhttps://github.com/YuvrajC7/sms-spam-detection')

page('1 Project scope and report guide')
h('Problem and objective')
p('Unwanted SMS messages often advertise prizes, offers or paid services. A useful classifier must recognize those patterns without treating every ordinary message containing a word such as “free” as spam. The objective is to build a working educational system, explain its probability calculations, measure its errors on unseen messages and demonstrate predictions interactively.')
p('The project follows the prescribed CountVectorizer and Multinomial Naive Bayes approach. The original messages are retained for traceability; the feature extractor performs its own lowercase tokenization. The system does not contact message senders, open links or automatically delete any SMS. Classification is performed locally.')
h('Report navigation')
table(['Section','Subject','Page'],[
('1','Scope and report guide','2'),('2','Dataset provenance and cleaning','3'),
('3','Class probabilities','4'),('4','Word probabilities and Bayes theorem','5'),
('5','Interpreting the five word events','6'),('6','Bernoulli random variable','7'),
('7','Data preparation and model','8'),('8','Evaluation results','9'),
('9','Error analysis','10'),('10','Live prediction and explanation','11'),
('11','Implementation and reproducibility','12'),('12','Validation and conclusions','13'),
('13','References and deliverable mapping','14')],[.6,5.4,.6])
h('Success criteria')
p('A successful submission must load and clean the UCI corpus, show class counts and probabilities, calculate all five requested conditional probabilities, derive expectation and variance, train on a stratified 80% subset, evaluate on the remaining 20% and accept a new SMS. It must also provide a notebook or Python source, a trained model, charts, a performance report and a GitHub repository link.')
p('This report is a separate submission artifact from the executable project. Tables and figures are generated from the saved analysis results so that the written evidence can be checked against the code. The repository address identifies the intended submission location; publication is a separate operational step from local execution.')

page('2 Dataset provenance and cleaning')
p('The source is the SMS Spam Collection by Almeida and Hidalgo, distributed through the UCI Machine Learning Repository [1]. Each source record contains a label followed by a tab and the message text. The downloadable source is saved locally, along with its original readme, so subsequent classroom runs do not require a network connection.')
h('Parsing and integrity')
p('Each nonblank source line is split at its first tab. This preserves quotation marks inside messages rather than interpreting them as CSV syntax. Labels are checked against the permitted values ham and spam. Missing fields and empty text are counted. The program rejects conflicting labels for the same cleaned message instead of silently choosing one. Every retained record stores its original source line and a message identifier.')
table(['Cleaning check','Observed result'],[
('Raw labelled records',m['audit']['raw_rows']),('Raw ham and spam','4,827 ham and 747 spam'),
('Missing label or message rows',0),('Empty messages after trimming',0),('Conflicting duplicate labels',0),
('Removed duplicate message rows',414),('Retained unique messages',5160),('Final ham and spam','4,518 ham and 642 spam')],[4.6,2])
h('Duplicate policy')
p('Leading and trailing whitespace is removed from message text before duplicate comparison. The first occurrence of each identical resulting message is retained. Duplicate removal is based on the message, not the entire row including its source line. Case differences and internal whitespace differences are preserved. Thus the policy is exact text deduplication after trimming, not semantic or template deduplication.')
p('Duplicate removal occurs before the train-test split. Otherwise an identical SMS could appear in both partitions and make the test score look stronger than it is. Near-duplicate promotional templates may still span the partitions; this remains a limitation of the evaluation.')
h('Identity and attribution')
p('The source archive is obtained from https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip. UCI lists the dataset under CC BY 4.0 [1]. The distributed data retains attribution in data/ATTRIBUTION.md. The source file checksum is recorded in results/metrics.json, enabling an exact byte-level source check.')
q=p('SHA256  '+m['audit']['sha256']); q.runs[0].font.size=Pt(8)
p('The assignment and UCI describe 5,574 instances, and the line-based parser observes the same total. Counts from other copies or CSV import settings may differ. The results in this report refer only to the exact saved source and cleaning policy above.')

page('3 Class probabilities')
p('Let S denote the event that a uniformly selected message from the cleaned corpus is spam, and H denote ham. These events are mutually exclusive and exhaustive because every retained message has one of the two labels. The probability estimates therefore sum to one.')
formula('P(S) = 642 / 5160 = 0.12441860')
formula('P(H) = 4518 / 5160 = 0.87558140')
table(['Class','Count','Probability','Percentage'],[
('Ham',4518,f'{m["p_ham"]:.8f}',f'{m["p_ham"]:.2%}'),
('Spam',642,f'{m["p_spam"]:.8f}',f'{m["p_spam"]:.2%}'),('Total',5160,'1.00000000','100.00%')],[1.5,1.3,2,1.8])
fig('class_distribution.png','Figure 1. Class counts after trimming and removing exact duplicate messages.',5.9)
h('Why class imbalance matters')
p('Ham accounts for about seven out of every eight cleaned messages. Accuracy alone can therefore be misleading: a classifier that predicts ham for everything already gets most messages correct while detecting no spam. The final evaluation includes spam precision, spam recall and F1 in addition to overall accuracy, and explicitly compares against an always-ham baseline.')
p('These empirical priors describe the cleaned corpus. They do not estimate the current spam prevalence in every phone user’s inbox. The learned model uses the training subset’s class frequencies, which are close to these proportions but not identical because the stratified split uses integer counts.')

page('4 Word probabilities and Bayes theorem')
p('For a suspicious word w, define W as the event that a message contains w as a whole word, ignoring letter case. The regular expression uses word boundaries on both sides. “FREE” matches “free”; “freedom” does not. Repeated occurrences in one message contribute once to this analysis. This event differs from the token-frequency features used by the classifier.')
formula('P(S | W) = n(S and W) / n(W)')
formula('P(S | W) = P(W | S) P(S) / P(W)')
table(['Word','All with word','Spam','Ham','Direct P(S | W)'],[
(r.word,r.containing,r.spam_containing,r.ham_containing,f'{r.p_spam_given_word:.6f}') for r in w.itertuples()],[1,1.4,.8,.8,2.6])
h('Worked example for free')
p('Among 199 messages containing “free”, 144 are spam and 55 are ham. The direct estimate is therefore 144/199. For Bayes’ theorem, the denominator P(Free) uses the entire cleaned corpus, while P(Free | Spam) uses only the 642 spam messages.')
formula('P(Free | S) = 144 / 642 = 0.22429907')
formula('P(Free) = 199 / 5160 = 0.03856589')
formula('P(S | Free) = (144/642)(642/5160) / (199/5160)')
formula('P(S | Free) = 144/199 = 0.72361809')
p('The intermediate counts cancel algebraically, so Bayes’ theorem and the direct ratio must agree. The implementation checks equality numerically for all five words. Calculation uses full precision; rounding is applied only for display.')
h('Avoiding a common reversal')
p('P(Spam | Free) is about 72.36%, whereas P(Free | Spam) is about 22.43%. The first asks how likely spam is among messages containing “free”; the second asks how often spam messages contain “free”. They use different sample spaces and must not be interchanged.')
p('The five words were prescribed by the assignment. Their full-corpus descriptive analysis is not fed into feature selection or model tuning. The trained classifier receives only training messages and labels. A word absent from the corpus would have an undefined conditional probability, not a probability of zero.')

page('5 Interpreting the five word events')
table(['Word','P(W | S)','P(W)','Bayes P(S | W)'],[
(r.word,f'{r.p_word_given_spam:.6f}',f'{r.p_word:.6f}',f'{r.bayes_verification:.6f}') for r in w.itertuples()],[1.1,1.8,1.7,2])
fig('word_probabilities.png','Figure 2. Conditional spam proportions for the five required words. The dashed line is the corpus spam prior.',5.9)
p('“Prize” occurs in 70 retained messages, all labelled spam, giving an observed proportion of 1.000000. This finite-sample result does not prove that every future message containing “prize” is spam. It also does not override other evidence in a full-message model prediction.')
p('“Urgent” appears in 61 messages, of which 56 are spam: 56/61 = 0.918033. “Win” appears in 53 messages, with 47 spam: 47/53 = 0.886792. “Offer” appears in 30 messages, with 24 spam: 24/30 = 0.800000. The different denominators matter because a proportion supported by fewer observations is less stable.')
p('“Free” provides a useful classroom counterexample. Its observed spam proportion is substantially above the 12.44% prior, yet 55 messages containing it are genuine. A phrase such as “Are you free for lunch tomorrow?” shows why a single-word rule is insufficient. The classifier combines evidence from all recognized tokens rather than using this five-word table as a rule list.')
p('The event counts are descriptive associations, not causal effects. The words do not cause the labels, and the dataset selection process influences the observed relationships. These tables quantify this corpus rather than establishing general laws of language.')

page('6 Bernoulli random variable analysis')
p('Define Y as an indicator of spam for a randomly selected cleaned message. Y takes only the values 0 and 1. A random variable with this two-outcome indicator structure follows a Bernoulli distribution under the empirical sampling model.')
formula('Y = 1 for spam and Y = 0 for ham')
formula('Y ~ Bernoulli(p), where p = 642/5160')
table(['Value y','Interpretation','P(Y = y)'],[('0','Ham','0.87558140'),('1','Spam','0.12441860')],[1,3.2,2.4])
h('Expected value')
formula('E(Y) = sum of y P(Y = y)')
formula('E(Y) = 0(1-p) + 1(p) = p = 0.12441860')
p('The expected value is the long-run fraction of spam under repeated uniform sampling from the empirical corpus. It does not mean that an individual message is “0.1244 spam”; each observed label is still either zero or one. The average of the 5,160 encoded labels equals this expectation exactly.')
h('Variance')
formula('Y squared = Y, so E(Y squared) = p')
formula('Var(Y) = E(Y squared) - [E(Y)] squared')
formula('Var(Y) = p - p squared = p(1-p) = 0.10893862')
p('Variance quantifies dispersion of the binary indicator around its mean. The product p(1-p) is largest at p=0.5 and becomes zero when all labels belong to one class. For this ham-dominated sample, the observed variance is below the maximum Bernoulli variance of 0.25.')
h('Verification and interpretation')
p('The notebook compares p(1-p) with the numerical variance of the encoded labels using ddof=0. This treats the cleaned corpus as the empirical distribution being described. Using the unbiased sample variance with ddof=1 would introduce a factor of n/(n-1) and would answer a different estimation question.')
p('As an illustration, 100 independent draws with replacement from this empirical distribution would have an expected spam count of 100p = 12.4419 and count variance 100p(1-p) = 10.8939. This extension requires independence and sampling with replacement; it is not a claim about a real sequence of 100 incoming messages.')

page('7 Data preparation and model building')
h('Stratified training and testing')
p('The code uses train_test_split with test_size=0.2, random_state=42 and stratify=target. Stratification approximately preserves the spam fraction in both partitions. The fixed random seed makes the split reproducible for the same ordered input and software environment. The saved split manifest records the membership explicitly.')
table(['Partition','Ham','Spam','Total'],[('Training',3614,514,4128),('Test',904,128,1032),('Total',4518,642,5160)],[2.2,1.4,1.4,1.6])
h('Text representation')
p('CountVectorizer is fitted only on training text and learns 7,673 vocabulary terms. Its default analyzer lowercases text and selects tokens containing at least two word characters. Punctuation acts as a separator. The output is a sparse nonnegative word-count matrix: each row is a message, each column is a vocabulary term and each value counts that term in the message. No stemming, stopword filtering or TF-IDF transformation is applied [2].')
p('The test messages are transformed with the already learned vocabulary. Unknown terms are ignored. A Pipeline stores the vectorizer and classifier together so live input undergoes the same transformation as training and test data. No vocabulary fitting is performed on the complete corpus before splitting.')
h('Multinomial Naive Bayes')
p('The classifier models word-count evidence for each class. Let N(c,j) be the total training count of vocabulary term j in class c, V be the vocabulary size and alpha=1. Laplace smoothing assigns a positive estimated probability even when a term has not appeared in one class [3].')
formula('theta(c,j) = [N(c,j) + 1] / [sum over k N(c,k) + V]')
formula('score(c) = log P(c) + sum over j x(j) log theta(c,j)')
p('The class prior is estimated from the training labels: P_train(Spam)=514/4128=0.12451550. The two class scores are normalized to produce predict_proba outputs. Computing in log space avoids multiplying many small probabilities directly. The predicted label is the class with the larger posterior score; a tie follows the library’s class order.')
p('“Naive” refers to the conditional independence assumption for token evidence given the class. This simplifies estimation but ignores word order and relationships such as negation. It is a suitable, interpretable baseline for the assigned discrete text features. The default model is used without test-driven tuning.')

page('8 Evaluation on held out messages')
p('All performance values below use the 1,032 test messages, which were excluded from fitting the vocabulary and classifier. Spam is the positive class. In Figure 3, rows are actual labels and columns are predicted labels, with ham first and spam second.')
fig('confusion_matrix.png','Figure 3. Confusion matrix on the fixed held-out test set.',4.6)
table(['Class or average','Precision','Recall','F1 score','Support'],[
(name,f'{m["classification_report"][key]["precision"]:.4f}',f'{m["classification_report"][key]["recall"]:.4f}',f'{m["classification_report"][key]["f1-score"]:.4f}',int(m['classification_report'][key]['support'])) for name,key in [('Ham','Ham'),('Spam','Spam'),('Macro average','macro avg'),('Weighted average','weighted avg')]],[2.2,1.1,1.1,1.2,1])
formula('Accuracy = (TN + TP) / N = (898 + 116)/1032 = 98.26%')
formula('Spam precision = TP/(TP+FP) = 116/122 = 95.08%')
formula('Spam recall = TP/(TP+FN) = 116/128 = 90.625%')
formula('Spam F1 = 2TP/(2TP+FP+FN) = 232/250 = 92.80%')
p('The always-ham baseline obtains 904/1032 = 87.60% accuracy and zero spam recall. The model improves accuracy by 10.66 percentage points and detects 116 of the 128 test spam messages. Macro averages weight the two classes equally; weighted averages weight them by their test supports. The latter are dominated by ham because it is more common.')

page('9 Understanding classification errors')
h('False negatives and false positives')
p('There are 12 false negatives: actual spam messages predicted as ham. They represent 12/128 = 9.375% of test spam. In a deployed filter, such messages would still reach the inbox. There are 6 false positives: genuine messages predicted as spam, corresponding to 6/904 = 0.664% of test ham. These errors could hide useful communications, so they are important even though they are numerically fewer.')
h('Examples from the held out errors')
errors=pd.read_csv(PROJECT/'results/errors.csv')
for line,comment in [
    (4863,'This genuine text mentions a phone brand. Brand and product terms can resemble promotional language. The model gives spam high probability despite the short ordinary statement.'),
    (3891,'The compact wording resembles a telecom offer. With little surrounding context, a bag-of-words model has limited evidence to distinguish conversation from advertising.'),
    (3361,'The labelled spam message imitates a personal conversation and a missed-call reply. Familiar conversational words can outweigh the suspicious contact pattern.'),
    (4257,'The labelled spam promotes a paid mobile game. Its particular vocabulary is not enough for this model to assign spam the larger probability. The training corpus does not guarantee coverage of every promotional style.')]:
    row=errors.loc[errors.source_line==line].iloc[0]
    h(f'Source line {line} and {row.error_type.lower()}')
    # Contact information is omitted in the report excerpt; source CSV remains intact.
    excerpt=row.message
    if line==3361: excerpt="Sorry I missed your call let's talk when you have the time. I'm on [number omitted]"
    p('Message excerpt: '+excerpt)
    p(f'Actual label: {row.label}. P(Spam | Message) = {row.p_spam:.6f}. '+comment)
h('What these examples establish')
p('These explanations describe plausible linguistic patterns, not experimentally proven causes. The exact model contribution of each recognized token can be inspected with the predictor. A confident error is possible because Naive Bayes probabilities are not necessarily calibrated. All 18 errors, their original source lines, labels and estimated spam probabilities are saved in results/errors.csv for audit.')
p('The error set is inspected after fitting to understand weaknesses. It is not used to change the model or choose a threshold in this submission. Future improvements should be selected on training-only validation data, reserving a fresh test set for final evaluation.')

page('10 Live prediction and explanation')
h('Demonstration result')
p('Input SMS: “Congratulations! You have won a free prize. Claim now.”')
formula('Prediction = Spam')
formula(f'P(Spam | Message) = {m["demo"]["p_spam"]:.10f}')
formula(f'P(Ham | Message) = {m["demo"]["p_ham"]:.10f}')
p('The two probabilities sum to one up to floating-point rounding. The assignment’s sample values of 0.96 and 0.04 are illustrative; the values above are computed from the saved fitted pipeline. The result is highly confident under this model, but it does not establish certainty that an arbitrary future message is spam.')
h('How the explanation is calculated')
p('For each known term, the interface displays its exact additive contribution to spam-versus-ham log odds: count × [log theta(Spam,word) - log theta(Ham,word)]. Positive values favor spam; negative values favor ham. The prior log odds must also be included. These contributions describe the model calculation, not causal influence.')
table(['Term','Count','Log odds contribution'],[(r['word'],r['count'],f"{r['log_odds']:+.4f}") for r in m['demo']['contributions']],[2.5,1,3.1])
p(f'The prior contributes {m["demo"]["prior_log_odds"]:.4f}. Adding all token contributions gives the final log odds, and the logistic transformation gives P(Spam | Message). The automated checks verify this reconstruction against predict_proba. The interface shows the strongest terms first; the command-line JSON exposes every term.')
h('Classroom interaction')
p('Run START_DEMO.cmd on Windows, then use the prize example, an ordinary library message, and “Are you free for lunch tomorrow?”. Enter an original message and select Analyse message, or press Ctrl+Enter. A second tab presents dataset counts, Bernoulli quantities, word probabilities and evaluation metrics.')
p('Empty input is rejected with a clear message. If no tokens are recognized, the interface warns that the probabilities reduce to the training priors. This is absence of usable vocabulary evidence, not a reliable judgement of safety. Input is limited to 10,000 characters to keep the classroom interface responsive.')

page('11 Implementation and reproducibility')
h('Project structure')
table(['File or folder','Purpose'],[
('spam_detector.py','Loading, cleaning, probabilities, training, evaluation and prediction'),
('demo.py','Offline desktop interface for classroom presentation'),
('SMS_Spam_Detection.ipynb','Executed notebook explaining all eight activities'),
('data/','Official source, attribution and cleaned CSV'),
('models/spam_pipeline.joblib','Fitted vectorizer and classifier in one pipeline'),
('results/','Metrics, charts, word calculations, split manifest and all test predictions'),
('tests/test_project.py','Automated correctness and consistency checks'),
('DEMO_GUIDE.md','Presentation sequence and viva preparation')],[2.5,4.1])
h('Reproduce the results')
p('Use Python 3.12 and the pinned requirements for the closest match to the recorded environment. From the project directory, install dependencies, run the training command and execute the tests. The saved source allows training to work offline once dependencies are available. If the source file is absent, the program downloads the official archive with a timeout.')
for line in ['python -m pip install -r requirements.txt','python spam_detector.py train','python -m unittest discover -s tests -v','python demo.py','python spam_detector.py interactive']:
    para=p(line); para.runs[0].font.name='Consolas'; para.runs[0].font.size=Pt(10)
p('The notebook can be opened in Jupyter or uploaded with the project files to Colab. Execute cells in order. To reproduce the published measurements exactly, preserve the source checksum, cleaning policy, ordered rows, split seed and dependency versions. The saved model is the same training-only model used in evaluation; it is not subsequently refitted on the full dataset.')
h('Recorded environment')
p('; '.join(f'{key} {value}' for key,value in m['versions'].items())+'.')
p('The model artifact uses joblib serialization. Only load the locally generated artifact or another trusted copy; model files are executable Python serialization, not a safe interchange format for unknown downloads. The project does not log new live-demo messages to disk.')

page('12 Validation limitations and conclusions')
h('Verification performed')
p('Six automated tests pass. They check the cleaning reconciliation, uniqueness and missing values; whole-word counting with a hand-verifiable fixture and Bayes equality; disjoint train/test membership and training-only vocabulary; probability normalization and exact log-odds reconstruction; empty, invalid and overlong input; and agreement between the saved model, saved test predictions and confusion matrix. The notebook is executed from start to finish, and the desktop callbacks are exercised with prize, ordinary, ambiguous and unknown-vocabulary inputs.')
h('Limits of the evidence')
p('The measured accuracy belongs to one fixed random split of a historical English SMS corpus. It does not establish present-day performance on multilingual messages, new scam techniques or a particular institution’s inbox. Exact duplicates are removed, but paraphrases and repeated campaign templates may remain. A group-based or time-based evaluation would better test some forms of generalization if appropriate metadata were available.')
p('The model ignores word order and treats token evidence as conditionally independent. Correlated words can lead to overconfident probabilities. No probability calibration is fitted, and no calibration guarantee is made. Tokens outside the training vocabulary are ignored; a message with no recognized tokens receives training priors. Extremely short messages can be difficult because they supply little context.')
p('The full-corpus word analysis serves the probability assignment, while performance evaluation uses a training-only model. A deployment-oriented analysis would estimate word statistics within training folds and evaluate their stability. Current false-negative and false-positive rates are descriptive test rates; no repeated-split uncertainty interval is claimed.')
h('Future work')
p('Reasonable next experiments include character n-grams for obfuscated spelling, bigrams for short phrases, training-only cross-validation for smoothing strength, and calibrated probabilities using held-out validation folds. Threshold selection should reflect the cost of missed spam versus blocked genuine messages. These are proposed improvements, not experiments performed in this submission.')
h('Conclusion and learning outcome')
p(f'The completed system links elementary probability with a functioning text classifier. Empirical class probabilities provide priors, word events illustrate conditional probability and Bayes’ theorem, and the spam indicator demonstrates Bernoulli expectation and variance. Word-count features and Multinomial Naive Bayes extend these ideas to complete messages. The held-out accuracy of {m["accuracy"]:.2%} is supported by a confusion matrix and class-specific metrics rather than accuracy alone.')
p('The executable demo, notebook, saved model and analysis outputs make the calculations inspectable during presentation. The project also provides a reproducible repository structure and attribution suitable for a public academic portfolio. Its strongest conclusion is that a simple probabilistic model performs well on this controlled dataset while still making identifiable, explainable errors.')

page('13 References and submission mapping')
h('References')
refs=[
'[1] Almeida, T. and Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5CC84. https://archive.ics.uci.edu/dataset/228/sms+spam+collection. Dataset license: CC BY 4.0.',
'[2] scikit-learn developers. CountVectorizer API reference. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html.',
'[3] scikit-learn developers. Naive Bayes user guide, including Multinomial Naive Bayes. https://scikit-learn.org/stable/modules/naive_bayes.html.',
'[4] scikit-learn developers. train_test_split API reference. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.',
'[5] Course assignment. BAMAT207 Probability and Statistics, Project Activity, Probabilistic SMS Spam Detection. Supplied Project_activity.docx. Submission deadline 27 October 2026.'
]
for ref in refs: p(ref)
p('Online documentation and dataset accessed on 26 October 2026. All reported numerical results are generated by this project from the saved UCI source; software documentation supports the method descriptions.')
h('Assignment deliverables')
table(['Required deliverable','Location in the submission'],[
('Python or Colab notebook','project/SMS_Spam_Detection.ipynb and spam_detector.py'),
('Cleaned dataset summary','Report sections 2 and 3; data/cleaned_sms.csv; results/metrics.json'),
('Class distribution chart','project/results/class_distribution.png'),
('Five suspicious word probabilities','Report sections 4 and 5; results/word_probabilities.csv'),
('Trained Naive Bayes model','project/models/spam_pipeline.joblib'),
('Confusion matrix and performance report','project/results/confusion_matrix.png and classification_report.txt'),
('Interactive SMS prediction','project/demo.py and command-line interactive mode'),
('Report in editable and final formats','report/SMS_Spam_Detection_Report.docx and .pdf'),
('GitHub submission address','https://github.com/YuvrajC7/sms-spam-detection')],[2.6,4])
h('Presentation readiness')
p('Before class, launch the desktop demo, read the worked Bayes example and verify the three central numbers: 5,160 cleaned messages, 642 spam messages and 1,032 held-out test messages. Be ready to distinguish empirical word probabilities from model posteriors, explain both error types and show why the always-ham baseline is insufficient. The separate demo guide provides a timed presentation sequence and short viva answers.')
for root in [doc.styles.element,doc.element,sec.footer._element]:
    for color in root.xpath('.//w:color'):
        color.attrib.clear()
        color.set(qn('w:val'),'auto')
doc.save(HERE/'SMS_Spam_Detection_Report.docx')
print('Report created:',HERE/'SMS_Spam_Detection_Report.docx')

