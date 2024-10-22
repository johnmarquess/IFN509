import numpy as np
import pandas as pd


def verbose_wrapper(func):
    """A wrapper to print function progress if verbose is True."""

    def wrapped_func(df, verbose=False):
        if verbose:
            print(f"Running {func.__name__}...")
        return func(df)

    return wrapped_func


# Redefine your functions to remove the verbose argument
def drop_columns_with_missing_values(df):
    """Drop specified columns with too many missing values."""
    columns_to_drop = ['A1Cresult', 'max_glu_serum']
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


# The rest of the functions remain the same without verbose
def recode_admission_type(df):
    adm_conditions = [
        df['admission_type_id'].isin([1, 2, 7]),
        df['admission_type_id'].isin([3, 4]),
        df['admission_type_id'].isin([5, 6, 8])
    ]
    choices = ['Emergency', 'Elective', 'Unknown']
    df['admission_type'] = np.select(adm_conditions, choices, default='Not available')
    return df


def recode_discharge_disposition(df):
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
    drug_order = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}
    drug_columns = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'insulin'
    ]
    for drug in drug_columns:
        df[drug] = df[drug].map(drug_order)
    return df


def recode_age_column(df):
    age_order = {
        '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3,
        '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7,
        '[80-90)': 8, '[90-100)': 9
    }
    df['age'] = df['age'].map(age_order)
    return df


def drop_deceased_patients(df):
    """Drop rows of data with deceased patients from the dataset."""
    return df[df['discharge_disposition'] != 'Deceased']


# Define the data_prep function that uses verbose_wrapper and a list of transformations
def data_prep(df, verbose=False):
    """Prepare the data by applying several transformation steps including dropping columns,
    recoding categories, and converting data types."""

    # List of transformation functions
    transformations = [
        drop_columns_with_missing_values,
        recode_medical_specialty,
        recode_admission_type,
        recode_discharge_disposition,
        recode_admission_source,
        recode_binary_values,
        recode_drug_columns,
        recode_age_column,
        drop_deceased_patients
    ]

    # Wrap and apply each transformation with verbose if enabled
    for transform in transformations:
        df = verbose_wrapper(transform)(df, verbose=verbose)

    if verbose:
        print("Data preparation complete.")

    return df
