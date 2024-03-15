import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

monthly_data_labelled = sys.argv[1]
monthly_data_unlabelled = sys.argv[2]
labels_output_csv = sys.argv[3]



def main():
    labelled_data = pd.read_csv(sys.argv[1])
    unlabelled_data = pd.read_csv(sys.argv[2])
    X = labelled_data.drop(['city', 'year'], axis=1)
    y = labelled_data['city']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    model_training_score=model.score(X_train,y_train)
    print('Validation score:', accuracy_score(y_val, y_pred))
    
    # df = pd.DataFrame({'truth': y_val, 'prediction': model.predict(X_val)})
    # print(df[df['truth'] != df['prediction']])
    
    unlabelled_data_processed = unlabelled_data.drop(['city', 'year'], axis=1)
    unlabelled_scaled = scaler.transform(unlabelled_data_processed)
    predictions = model.predict(unlabelled_scaled)
    pd.Series(predictions).to_csv(labels_output_csv, index=False, header=False)
    
if __name__ == '__main__':
    main()