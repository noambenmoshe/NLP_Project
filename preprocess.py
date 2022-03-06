import os
import pandas as pd
from deep_translator import GoogleTranslator

def combine_text(df, col_a = 'text a', col_b ='text b'):
    def concat_lines(text_a, text_b):
        if pd.isna(text_a) or text_a == 0 or text_a == ' ':
            text = ''
        else:
            text = text_a
        if not (pd.isna(text_b) or text_b == 0 or text_b == ' '):
            text = text + text_b

        return text

    df['Text'] = df.apply(lambda x: concat_lines(text_a=x[col_a], text_b=x[col_b]), axis=1)
    df = df.drop(df[df.Text == ''].index)
    return df


def load_raw_data(translate=True, val_set_size=100):

    path_to_rbdf = '/MLAIM/AIMLab/Shany/databases/rbafdb/'
    main_path = '/home/b.noam/NLP_Project/'
    path_to_dictionary = path_to_rbdf + 'documentation/reports/RBAF_reports.xlsx'
    path_to_validation = main_path + 'validation_set.xlsx'
    path_to_test = main_path + 'test_set.xlsx'
    out_dir = '/home/b.noam/NLP_Project/preprocessed_data/'

    train_set = pd.read_excel(path_to_dictionary, engine='openpyxl')
    validation_set = pd.read_excel(path_to_validation, engine='openpyxl')
    test_set = pd.read_excel(path_to_test, engine='openpyxl')

    test_set_ids = test_set['holter_id'].tolist()
    validation_set_ids = validation_set['holter_id'].tolist()

    labeled_data_ids = validation_set_ids[val_set_size:]
    validation_set_ids = validation_set_ids[:val_set_size]


    train_set = combine_text(train_set, col_a='טקסט מתוך סיכום', col_b='טקסט מתוך תוצאות')
    validation_set = combine_text(validation_set)
    test_set = combine_text(test_set)

    # remove test_set ids from the training set
    train_set = train_set[~train_set['holter_id'].isin(test_set_ids)]
    # remove validation_set ids from the training set
    train_set = train_set[~train_set['holter_id'].isin(validation_set_ids)]

    # remove labeled data ids from validation
    val_set = validation_set[validation_set['holter_id'].isin(validation_set_ids)]
    # create a labled data set (all of the validation set that is not part of the validaion set)
    labeled_data = validation_set[validation_set['holter_id'].isin(labeled_data_ids)]

    if translate:
        data_frames = {
                        'train_set': train_set,
                       'val_set': val_set,
                      'test_set': test_set,
                      'labeled_data': labeled_data
                       }
        translator = GoogleTranslator(source='auto', target='en')

        def _translate_line(text):
            # mange progress counter
            if (hasattr(_translate_line, "counter") and hasattr(_translate_line, "total_idx")):
                if _translate_line.counter % 100 == 0:
                    print("translate line {}/{}".format(_translate_line.counter, _translate_line.total_idx))
                _translate_line.counter += 1

            # apply translate
            return translator.translate(text)

        for k, df in data_frames.items():
            print('translate: ', k)
            _translate_line.counter = 0
            _translate_line.total_idx = len(df)
            df['text_en'] = df.apply(lambda x: _translate_line(text=x['Text']), axis=1)


    for k, df in data_frames.items():
        df.to_pickle(os.path.join(out_dir, f"{k}.pkl"))


    return data_frames

def load_preprocessed(preprocessed_dir):
    # data_dir = '/home/b.noam/NLP_Project/preprocessed_data/'
    data_frames_keys = [
        'train_set',
       'val_set',
        'test_set',
        'labeled_data'
    ]
    data_frames = {}
    for k in data_frames_keys:
        data_frames[k] = pd.read_pickle(os.path.join(preprocessed_dir, f"{k}.pkl"))

    return data_frames


if __name__ == '__main__':
    data_frames_0 = load_raw_data()
    data_frames_load = load_preprocessed()
    print("done")
    