from data_management import *
import matplotlib.pyplot as plt

df = get_ling_feats(_set='train')
#_dict = {col: 0 for col in list(df.columns)}
for th in [0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1]:
    df2 = filter_features(df, th=th, verbose=False)
    cols = list(df2.columns)
    print(th, len(cols))