# train.py
import re
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import tldextract
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def has_ip(url):
    # simple check for IPv4 in domain
    m = re.search(r'//(\d{1,3}(?:\.\d{1,3}){3})', url)
    return 1 if m else 0

def count_digits(s):
    return sum(c.isdigit() for c in s)

def extract_features(url):
    u = url.strip()
    parsed = urlparse(u)
    domain = parsed.netloc
    path = parsed.path or ""
    ext = tldextract.extract(u)
    subdomain = ext.subdomain
    domain_only = ext.domain

    features = {}
    features['url_len'] = len(u)
    features['domain_len'] = len(domain)
    features['path_len'] = len(path)
    features['count_dots'] = domain.count('.')
    features['count_hyphen'] = domain.count('-') + path.count('-')
    features['has_at'] = 1 if '@' in u else 0
    features['has_https'] = 1 if parsed.scheme == 'https' else 0
    features['has_ip'] = has_ip(u)
    features['count_digits'] = count_digits(u)
    # suspicious tokens
    suspicious = ['secure', 'account', 'webscr', 'login', 'update', 'bank', 'confirm', 'verify', 'paypal', 'signin']
    features['suspicious_tokens'] = sum(1 for t in suspicious if t in u.lower())
    # number of subdomain parts
    features['subdomain_parts'] = 0 if not subdomain else subdomain.count('.') + 1
    return features

def df_features(df):
    feats = df['url'].apply(extract_features).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

def main():
    df = pd.read_csv('data.csv')  # assumes columns 'url' and 'label' (1 phishing, 0 legit)
    df = df.dropna(subset=['url','label'])
    data = df_features(df)
    X = data[['url_len','domain_len','path_len','count_dots','count_hyphen',
              'has_at','has_https','has_ip','count_digits','suspicious_tokens','subdomain_parts']]
    y = data['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # quick model - Logistic Regression. If time, try RandomForest.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))

    joblib.dump(model, 'model.joblib')
    print("Saved model.joblib")

if __name__ == '__main__':
    main()