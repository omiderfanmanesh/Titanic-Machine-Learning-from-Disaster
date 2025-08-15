import re
import unittest

import numpy as np
import pandas as pd


class MyTestCase(unittest.TestCase):
    def extract_title(name: str) -> str:
        m = re.search(r',\s*([^\.]+)\.', str(name))
        return m.group(1).strip() if m else 'Unknown'

    def do_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Title'] = df['Name'].map(self.extract_title)
        title_map = {
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
            'Lady': 'Royal', 'Countess': 'Royal', 'Dona': 'Royal', 'Sir': 'Royal', 'Don': 'Royal',
            'Jonkheer': 'Rare', 'Capt': 'Rare', 'Col': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare'
        }
        df['Title'] = df['Title'].replace(title_map)
        df['Surname'] = df['Name'].str.split(',').str[0]
        df['Title_First_Middle'] = df['Name'].str.split(',').str[1].str.strip()
        df['Title_Raw'] = df['Title_First_Middle'].str.split(' ').str[0]
        df['First_Middle'] = df['Title_First_Middle'].str.split(' ').str[1:].str.join(' ')
        df['First_Middle'] = df['First_Middle'].replace('', pd.NA)  # replace empty strings with NaN
        df['First_Middle'] = df['First_Middle'].fillna('Unknown')  # fill NaN with 'Unknown'
        df['Surname'] = df['Surname'].str.strip()  # remove leading and trailing spaces
        df['Title_Raw'] = df['Title_Raw'].str.replace('.', '', regex=False).str.strip()  # clean raw title
        df['First_Middle'] = df['First_Middle'].str.strip()  # remove leading and
        df['MaidenName'] = df['Name'].str.extract(r'\((.*?)\)')
        df['ZeroFare'] = (df['Fare'] == 0).astype(int)
        # Extract Ticket prefix
        df['Ticket_prefix'] = (
            df['Ticket']
            .astype(str)
            .str.replace(r'\d+', '', regex=True)  # remove digits
            .str.replace('.', '', regex=False)  # remove dots
            .str.strip()  # trim spaces
        )

        # Replace empty prefixes with 'NUMBER'
        df['Ticket_prefix'] = df['Ticket_prefix'].replace('', 'NUMBER')

        # Extract numeric part of the Ticket
        df['Ticket_number'] = (
            df['Ticket']
            .astype(str)
            .str.extract(r'(\d+)$')[0]  # extract last group of digits
            .astype(float)  # convert to numeric
        )

        # Deck from Cabin (first letter); many missing
        df['Deck'] = df['Cabin'].astype(str).str[0]
        df['Deck'] = df['Deck'].where(df['Deck'].isin(list('ABCDEFGT')), other='U')

        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        ticket_counts = df['Ticket'].value_counts()
        df['TicketGroupSize'] = df['Ticket'].map(ticket_counts)

        return df

    def do_fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        # ===== 1. Fill missing 'Embarked' with mode =====
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

        # ===== 2. Fill missing 'Fare' in test with median by Pclass & Embarked =====
        df['Fare'] = df.groupby(['Pclass', 'Embarked'])['Fare'] \
            .apply(lambda x: x.fillna(x.median()))

        # ===== 3. Fill missing 'Age' using median of Title & Pclass =====
        df['Age'] = df.groupby(['Title', 'Pclass'])['Age'] \
            .apply(lambda x: x.fillna(x.median()))

        # ===== 4. Fill missing 'Deck' with 'U' (Unknown) =====
        df['Deck'] = df['Deck'].fillna('U')

        # ===== 5. Fill missing 'Cabin' with 'Unknown' =====
        df['Cabin'] = df['Cabin'].fillna('Unknown')

        # ===== 6. Fill missing 'Ticket_prefix' with 'NONE' =====
        df['Ticket_prefix'] = df['Ticket_prefix'].fillna('NONE')

        # ===== 7. Fill missing 'Ticket_number' with -1 =====
        df['Ticket_number'] = df['Ticket_number'].fillna(-1)

        # ===== 8. Fill any remaining NaN in categorical columns with 'Unknown' =====
        cat_cols = df.select_dtypes(include='object').columns
        df[cat_cols] = df[cat_cols].fillna('Unknown')

        # ===== 9. Fill any remaining NaN in numeric columns with median =====
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        return df

    def test_something(self):
        # Import necessary libraries

        # Load the dfset
        df_train = pd.read_csv('../data/raw/train.csv')
        df_test = pd.read_csv('../data/raw/test.csv')

        df_train_new = self.do_preprocessing(df_train)
        df_test_new = self.do_preprocessing(df_test)

        df_train_new_no_missing_values = self.do_fill_missing(df_train_new)
        df_test_new_no_missing_values = self.do_fill_missing(df_test_new)



if __name__ == '__main__':
    unittest.main()
