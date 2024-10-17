import numpy as np
import pandas as pd


def drop_columns_with_missing_values(df, columns_to_drop):
    """Drop specified columns with too many missing values."""
    return df.drop(columns_to_drop, axis=1)


def recode_medical_specialty(df):
    """Recode the 'medical_specialty' column into fewer categories."""
    specialty = df['medical_specialty'].copy()

    specialty_conditions = [
        specialty.str.contains('Surg|Anesth|Ortho', case=False, regex=True),
        specialty.str.contains('Family|Outreach', case=False, regex=True),
        specialty.str.contains('Internal|Neph|Urol|Card|nfect|Pulmo|Endoc|Gastro|Hospital|Diag|Obs|Gyn|Haem|Onc',
                               case=False, regex=True),
        specialty.str.contains('Emerg|Trauma', case=False, regex=True),
        specialty == '?'
    ]

    specialty_choices = ['Surgical', 'General Practice', 'Internal Medicine', 'Emergency', 'Invalid']

    df['medical_specialty'] = np.select(specialty_conditions, specialty_choices, default='Other')
    return df


def recode_admission_type(df):
    """Recode the 'admission_type_id' column into fewer categories."""
    adm_conditions = [
        df['admission_type_id'].isin([1, 2, 7]),
        df['admission_type_id'].isin([3, 4]),
        df['admission_type_id'].isin([5, 6, 8])
    ]

    choices = ['Emergency', 'Elective', 'Unknown']
    df['admission_type'] = np.select(adm_conditions, choices, default='Not available')
    return df


def recode_discharge_disposition(df):
    """Recode the 'discharge_disposition_id' column into fewer categories."""
    dis_conditions = [
        df['discharge_disposition_id'].isin(
            [2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 22, 23, 24, 27, 28, 29, 30]),
        df['discharge_disposition_id'].isin([1]),
        df['discharge_disposition_id'].isin([7, 18, 25, 26]),
        df['discharge_disposition_id'].isin([11, 19, 20, 21])
    ]

    choices = ['AdditonalCare', 'Home', 'Unknown', 'Deceased']
    df['discharge_disposition'] = np.select(dis_conditions, choices, default='Not available')
    return df


def recode_admission_source(df):
    """Recode the 'admission_source_id' column into fewer categories."""
    source_conditions = [
        df['admission_source_id'].isin([1, 2, 3]),
        df['admission_source_id'].isin([4, 5, 6, 10, 22, 25]),
        df['admission_source_id'].isin([7, 8]),
        df['admission_source_id'].isin([9, 15, 17, 20, 21])
    ]

    choices = ['Referral', 'Transfer', 'Emergency', 'Unknown']
    df['admission_source'] = np.select(source_conditions, choices, default='Not available')
    df.drop(['admission_type_id', 'discharge_disposition_id', 'admission_source_id'], axis=1, inplace=True)
    return df


def recode_binary_values(df):
    df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
    df['change'] = df['change'].astype(int)
    df['diabetesMed'] = df['diabetesMed'].astype(int)
    return df


def recode_drug_columns(df):
    """
    Recode drug-related columns according to the specified order.
    """
    drug_order = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}

    # List of columns to be mapped
    drug_columns = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'insulin'
    ]

    # Apply mapping to each column
    for drug in drug_columns:
        df[drug] = df[drug].map(drug_order)

    return df


def recode_age_column(df):
    """
    Recode the 'age' column according to the specified age ranges.
    """
    age_order = {
        '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3,
        '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7,
        '[80-90)': 8, '[90-100)': 9
    }

    # Apply mapping to the 'age' column
    df['age'] = df['age'].map(age_order)

    return df


def data_prep(df):
    """Prepare the data by applying several transformation steps including dropping columns,
    recoding categories, and converting data types."""
    # Step 1: Drop unnecessary columns
    df = drop_columns_with_missing_values(df, ['A1Cresult', 'max_glu_serum'])

    # Step 2: Recode medical_specialty
    df = recode_medical_specialty(df)

    # Step 3: Recode admission_type_id
    df = recode_admission_type(df)

    # Step 4: Recode discharge_disposition_id
    df = recode_discharge_disposition(df)

    # Step 5: Recode admission_source_id
    df = recode_admission_source(df)

    # Step 6: Recode binary columns
    df = recode_binary_values(df)

    # Step 7: Recode drug columns
    df = recode_drug_columns(df)

    # Step 8: Recode age column
    df = recode_age_column(df)

    return df
