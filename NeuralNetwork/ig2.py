
from utils import *
import numpy as np
import tensorflow as tf
from data_management import *
import matplotlib.pyplot as plt
from alibi.explainers import IntegratedGradients

print('Hello World! This script is to evaluate feature importance of the neural network with Integrated Gradients :)')
result_path, features, _, _ = parse()
model = tf.keras.models.load_model(result_path+'/best') 
ig = IntegratedGradients(
    model, 
    layer=None,  
    method="gausslegendre", 
    n_steps=50, 
    internal_batch_size=500
)

th = -1000
if features == 'ling':
    df_train, df_test = get_ling_feats()
    th = 0.005
else:
    df_train, _, _ = load_dataframe(_set='train', mode='full', exclude_modality='ht', only_numeric=True)
    df_test, _, _ = load_dataframe(_set='test', mode='mask_subject', exclude_modality='ht', only_numeric=True, verbose=True)
    if features == 'both':
        df_train2, df_test2 = get_ling_feats() 
        df_train2.index = df_train.index
        df_train = pd.concat([df_train, df_train2], axis='columns')
        df_test2.index = df_test.index
        df_test = pd.concat([df_test, df_test2], axis='columns')
    th = 0.005

columns = list(df_test.columns)
input_len = len(columns)
print(columns)

X_train, X_test = scaling(df_train, df_test)
predictions = model.predict(X_test).argmax(axis=1) 
explanation = ig.explain(X_test, target=predictions)
attributions = explanation.attributions
print(len(attributions), len(attributions[0]), len(attributions[0][0]))
avg = np.mean(attributions[0], axis=0)
stds = np.std(attributions[0], axis=0)  
over_th = avg > th
avg = avg[over_th]
stds = stds[over_th]
columns = np.array(columns)[over_th]
print(len(columns))
plt.figure(figsize=(20, 10))
plt.title("IG-based Feature Importances")
plt.barh(range(avg.shape[0]), avg, color="r", xerr=stds, align="center")
# If you want to define your own labels,
# change indices to a list of labels on the following line.
plt.yticks(range(avg.shape[0]), columns)
plt.ylim([-1, avg.shape[0]])
plt.show()
plt.savefig(result_path + '/ig_importances.png')
