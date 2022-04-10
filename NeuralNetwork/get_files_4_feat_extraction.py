from data_management import *
import os
df_train, _, _ = load_dataframe(_set='train', mode='full', exclude_modality='ht', only_numeric=False, verbose=True)
df_test, _, _ = load_dataframe(_set='test', mode='mask_subject', exclude_modality='ht', only_numeric=False, verbose=True)
columns = list(df_test.columns)
print(columns)
input_len = len(columns)
print(input_len)

def get_files(texts, filename):
    for i, text in enumerate(texts):
        with open(filename % i, 'w') as txtfile:
            txtfile.write(text)

def mkdirs(dir_path, verbose=True):
  try:
    os.makedirs(dir_path)
  except OSError:
    if verbose:
      print ("Creation of the directory %s failed" % dir_path)

for x in ['mt_text', 'tgt_text']:
    folder = './train_' + x
    mkdirs(folder)
    texts = list(df_train[x])
    get_files(texts, folder + '/%03d.txt')

for x in ['mt_text', 'tgt_text']:
    folder = './test_' + x
    mkdirs(folder)
    texts = list(df_test[x])
    get_files(texts, folder + '/%03d.txt')