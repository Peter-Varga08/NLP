import data_management
import utils

df_train, *_ = data_management.load_dataframe(_set='train', mode='full', exclude_modality='ht', only_numeric=False, verbose=True)
df_test, *_ = data_management.load_dataframe(_set='test', mode='mask_subject', exclude_modality='ht', only_numeric=False,
                               verbose=True)

for df, name in zip([df_train, df_test], ['train', 'test']):
    for x in ['mt_text', 'tgt_text']:
        folder = f'./{name}_' + x
        utils.mkdirs(folder)
        texts = list(df[x])
        utils.get_files(texts, folder + '/%03d.txt')
