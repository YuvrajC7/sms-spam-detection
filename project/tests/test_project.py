import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from spam_detector import ROOT, load_data, predict, word_probabilities

class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(ROOT/'models/spam_pipeline.joblib')
        cls.df, cls.audit = load_data()
        cls.metrics = json.loads((ROOT/'results/metrics.json').read_text())

    def test_cleaning(self):
        self.assertFalse(self.df.message.duplicated().any())
        self.assertFalse(self.df[['message','label']].isna().any().any())
        self.assertEqual(self.audit['raw_rows']-self.audit['missing_rows']-self.audit['blank_rows']-self.audit['duplicate_rows'],len(self.df))

    def test_word_events_and_bayes(self):
        fixture=pd.DataFrame({'message':['FREE free!', 'free lunch', 'freedom', 'a prize'], 'target':[1,0,0,1]})
        rows=word_probabilities(fixture).set_index('word')
        self.assertEqual(rows.loc['free','containing'],2)
        self.assertEqual(rows.loc['free','p_spam_given_word'],.5)
        real=word_probabilities(self.df)
        np.testing.assert_allclose(real.p_spam_given_word,real.bayes_verification,atol=1e-14)

    def test_split_and_vocabulary_leakage(self):
        split=pd.read_csv(ROOT/'results/split_manifest.csv')
        train=set(split.loc[split.partition=='train','message_id'])
        test=set(split.loc[split.partition=='test','message_id'])
        self.assertFalse(train & test)
        self.assertEqual(len(train)+len(test),len(self.df))
        analyzer=self.model.named_steps['vectorizer'].build_analyzer()
        training_words={word for text in self.df.loc[self.df.message_id.isin(train),'message'] for word in analyzer(text)}
        self.assertEqual(training_words,set(self.model.named_steps['vectorizer'].vocabulary_))

    def test_predictions_and_log_odds(self):
        for message in ['Congratulations! You have won a free prize. Claim now.','See you at the library','zzzxqv qqqzzz','!!!']:
            result=predict(message,self.model)
            self.assertAlmostEqual(result['p_spam']+result['p_ham'],1)
            score=result['prior_log_odds']+sum(row['log_odds'] for row in result['contributions'])
            self.assertAlmostEqual(result['p_spam'],1/(1+np.exp(-score)),places=10)
        result=predict('zzzxqv qqqzzz',self.model)
        self.assertTrue(result['warning'])
        self.assertAlmostEqual(result['p_spam'],self.metrics['train_counts']['spam']/self.metrics['train_size'])

    def test_input_validation(self):
        for value in ['', '  ',None,'x'*10001]:
            with self.assertRaises(ValueError): predict(value,self.model)

    def test_saved_evaluation_matches_model(self):
        test=pd.read_csv(ROOT/'results/test_predictions.csv')
        predictions=self.model.predict(test.message)
        np.testing.assert_array_equal(predictions,test.prediction)
        np.testing.assert_array_equal(confusion_matrix(test.target,predictions),self.metrics['confusion_matrix'])
        self.assertEqual(int((test.error_type!='Correct').sum()),int((test.target!=predictions).sum()))

if __name__=='__main__': unittest.main(verbosity=2)
