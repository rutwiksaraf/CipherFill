import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import string
from sklearn.metrics import precision_score, recall_score, accuracy_score

def extract_features(df, is_test=False):
    features = []
    
    for _, row in df.iterrows():
        context = row['context']
        
        if not isinstance(context, str):
            context = str(context)  # Convert to string if it's not a string
        
        if not is_test:
            name = row['name']
            redacted_length = len(name)  # Redacted name length
        else:
            redacted_length = 10  
        
        n_grams = extract_ngrams(context, redacted_length)
        
        previous_word = get_previous_word(context)
        next_word = get_next_word(context)
        
        num_letters = redacted_length
        num_spaces = context.count(' ')  # Space count in context
        
        features.append({
            'unigrams': n_grams['unigrams'],
            'bigrams': n_grams['bigrams'],
            'trigrams': n_grams['trigrams'],
            'previous_word': previous_word,
            'next_word': next_word,
            'num_letters': num_letters,
            'num_spaces': num_spaces
        })
    
    # Convert to dataframe
    features_df = pd.DataFrame(features)
    return features_df



def extract_ngrams(text, redacted_length):
    words = text.split()
    
    unigrams = ""
    bigrams = ""
    trigrams = ""
    
    start_index = text.find('██████████')
    
    if start_index != -1:
        start_word = max(0, len(text[:start_index].split()) - 1)
        end_word = start_word + redacted_length  # Assuming redacted word length is number of █ characters
        
        unigrams = ' '.join(words[start_word:end_word])
        bigrams = ' '.join([f"{words[i]} {words[i+1]}" for i in range(start_word, len(words)-1)])
        trigrams = ' '.join([f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(start_word, len(words)-2)])
    
    return {'unigrams': unigrams, 'bigrams': bigrams, 'trigrams': trigrams}


# Get previous word
def get_previous_word(context):
    words = context.split()
    for i, word in enumerate(words):
        if '████' in word:
            return words[i-1] if i-1 >= 0 else ""
    return ""

# Get next word
def get_next_word(context):
    words = context.split()
    for i, word in enumerate(words):
        if '████' in word:
            return words[i+1] if i+1 < len(words) else ""
    return ""

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['split', 'name', 'context'], on_bad_lines='skip', engine='python')
    df.dropna(subset=['split', 'name', 'context'], inplace=True)  # Ensure no NaN rows
    return df

def train_model(train_df, val_df):
    train_features = extract_features(train_df)
    val_features = extract_features(val_df)
    
    train_labels = train_df['name']
    val_labels = val_df['name']
    
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_features['unigrams'] + " " + train_features['bigrams'] + " " + train_features['trigrams'])
    X_val = vectorizer.transform(val_features['unigrams'] + " " + val_features['bigrams'] + " " + val_features['trigrams'])
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, train_labels)
    
    val_preds = rf.predict(X_val)
    
    precision = precision_score(val_labels, val_preds, average='weighted')
    recall = recall_score(val_labels, val_preds, average='weighted')
    accuracy = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average='weighted')
    
    print("Validation Classification Report:")
    print(classification_report(val_labels, val_preds))
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return rf, vectorizer

def predict_and_generate_submission(model, vectorizer, test_file, output_file='submission.tsv'):
    test_df = pd.read_csv(test_file, sep='\t', header=None, names=['id', 'context'])
    
    test_features = extract_features(test_df, is_test=True)
    
    X_test = vectorizer.transform(test_features['unigrams'] + " " + test_features['bigrams'] + " " + test_features['trigrams'])
    
    test_preds = model.predict(X_test)
    
    submission_df = pd.DataFrame({'id': test_df['id'], 'name': test_preds})
    submission_df.to_csv(output_file, index=False, sep='\t')


# Main function to run the pipeline
def main():
    train_df = load_data('unredactor.tsv')  # Training data path
    val_df = train_df[train_df['split'] == 'validation']
    train_df = train_df[train_df['split'] == 'training']
    
    model, vectorizer = train_model(train_df, val_df)
    
    predict_and_generate_submission(model, vectorizer, 'test.tsv', 'submission.tsv')

if __name__ == "__main__":
    main()
