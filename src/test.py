# import argparse
#
#
# def test_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-te", "--test_file", default='./data/IK_NLP_22_PESTYLE/test.tsv',
#                         type=str, help="Location of test file.")
#     parser.add_argument("-r", "--result_path", default='./transformer_predictions.txt',
#                         help="Directory where to save results.")
#
#     # Subparsers to allow the different kind of model loaders be used without conflicts
#     subparsers = parser.add_subparsers()
#     bert_parser = subparsers.add_parser("BERT")
#     bert_parser.add_argument("-m", "--bert_model", type=str, help="Specify path to an already fine-tuned HuggingFace "
#                                                                   "BERT model.")
#     nn_parser = subparsers.add_parser("NN")
#     nn_parser.add_argument("-m", "--nn_model", type=str,
#                               help="Specify path to an already fine-tuned PyTorch NN model.")
#     ml_parser = subparsers.add_parser('ML')
#     ml_parser.add_argument('-m', '--sklearn_model', type=str,
#                            help="Specify path to an already fine-tuned sklearn model.")
#
#
# # TODO
# # TRANSFORMER
# # --------------------------------------------------------------------------------------------------------------------
# args = create_arg_parser()
# all_predictions = []
#
# if args.model:  # Use an already fine-tuned model, test on the final test set
#     model = ClassificationModel("bert", args.model, use_cuda=False)
#     predictions = predict_test(args.test_file, model, final=True)
#     all_predictions.append(predictions)
#     all_predictions.append("\n")
#
# # Write results to pickle file
# with open(args.outfile, 'w') as pred_file:
#     for pred_list in all_predictions:
#         for pred in pred_list:
#             pred_file.write(str(pred))
# print("Predictions have been written to file: " + args.outfile)
#
# # TODO
# # NEURAL NETWORK
# # --------------------------------------------------------------------------------------------------------------------
# args = utils.parse_nn_model()
# model = tf.keras.models.load_model(args.result_path + "/best")
#
# if args.features == "ling":
#     df_train, df_test = utils.get_ling_feats()
# else:
#     df_train = utils.load_dataframe(
#         split=Split.TRAIN, mode=ConfigMode.FULL, exclude_modality=Modality.SCRATCH, only_numeric=True
#     )
#     df_test = utils.load_dataframe(
#         split=Split.TEST,
#         mode=ConfigMode.MASK_SUBJECT,
#         exclude_modality=Modality.SCRATCH,
#         only_numeric=True,
#     )
#     if args.features == "both":
#         df_train_ling, df_test_ling = utils.get_ling_feats()
#         df_train_ling.index, df_test_ling.index = df_train.index, df_test.index
#         df_train = pd.concat([df_train, df_train_ling], axis="columns")
#         df_test = pd.concat([df_test, df_test_ling], axis="columns")
#
# X_train, X_test = utils.scaling(df_train, df_test)
# predictions = model.predict(X_test).argmax(axis=1)
#
# with open(args.result_path + "/predictions_both.txt", "w") as txt:
#     for pred in predictions:
#         txt.write(pred)
#
#     # TODO
#     # STATISTICAL MODELS
#
#     # |-------------------------|
#     # |     Load datasets       |
#     # |-------------------------|
#     logger.info("Loading datasets...")
#     df_train: pd.DataFrame = load_dataset("GroNLP/ik-nlp-22_pestyle", "full",
#                                           data_dir='../IK_NLP_22_PESTYLE')['train'].to_pandas()
#     df_test: pd.DataFrame = load_dataset("GroNLP/ik-nlp-22_pestyle", "mask_subject",
#                                          data_dir='../IK_NLP_22_PESTYLE')['test'].to_pandas()
#     df_train = df_train[df_train.modality != 'ht']
#     df_test = df_test[df_test.modality != 'ht']
#     y_train = np.array(df_train.subject_id)
#     y_test = pd.read_csv('../Statistical_and_ML_models/test_ground_truths.csv', header=None).to_numpy()
#     # one label-encoder is enough, categories are the same for both datasets
#     assert np.array_equal(np.unique(y_train),
#                           np.unique(y_test)), "Unique labels within train and test set are different."
#     label_encoder = LabelBinarizer().fit(y_train)
#     y_train_encoded = label_encoder.transform(y_train)  # used for 'run_test()' only
#
#     # Numeric features
#     logger.info("Creating numeric features.")
#     X_train_numeric = select_columns(df_train, dtype_include=['float32', 'int32'], name_exclude=['item_id'])
#     X_test_numeric = select_columns(df_test, dtype_include=['float32', 'int32'], name_exclude=['item_id'])
#     # Linguistic features
#     logger.info("Loading linguistic features...")
#
#     lingfeat = [
#         LingFeatDF(pd.read_csv('../linguistic_features/train_mt.csv',
#         sep="\t").drop(columns=['Filename']), 'train_mt'),
#         LingFeatDF(pd.read_csv('../linguistic_features/test_mt.csv', sep='\t').drop(columns=['Filename']), 'test_mt'),
#         LingFeatDF(pd.read_csv('../linguistic_features/train_tgt.csv', sep="\t").drop(columns=['Filename']),
#                    'train_tgt'),
#         LingFeatDF(pd.read_csv('../linguistic_features/test_tgt.csv',
#         sep="\t").drop(columns=['Filename']), 'test_tgt'),
#     ]
#     intersected_linguistic_columns = intersect_linguistic_columns(*[nt.df for nt in lingfeat])
#     col_lens = []
#     for idx, nt in enumerate(lingfeat):
#         lingfeat[idx] = filter_linguistic_columns(nt.df, intersected_linguistic_columns)
#         col_lens.append(lingfeat[idx].df.columns)
#     assert len(set(col_lens)) == 1, "Number of columns have to be equal for all linguistic feature dataframes."
#
#     X_train_ling = subtract_df(lingfeat[0], lingfeat[2])
#     X_test_ling = subtract_df(lingfeat[1], lingfeat[3])
#     # Combined features
#     X_train_combined = pd.concat([X_train_numeric.reset_index(drop=True),
#     X_train_ling.reset_index(drop=True)], axis=1)
#     X_test_combined = pd.concat([X_test_numeric.reset_index(drop=True), X_test_ling.reset_index(drop=True)], axis=1)
#
#     # |-------------------------|
#     # |       Load models       |
#     # |-------------------------|
#     if args.model_type == 'lr':
#         estimator = LinearRegression(n_jobs=-1)
#     elif args.model_type == 'rf':
#         estimator = RandomForestClassifier(n_jobs=-1, random_state=42)
#     else:
#         raise ValueError('Invalid model type.')
#
#     # |-------------------------|
#     # |     Run experiments     |
#     # |-------------------------|
#     if args.experiment_type == 'numeric':
#         if not args.test:
#             run_numeric_data_experiments(estimator, X_train_numeric, y_train, scaling=args.scaling)
#             estimator.fit(X_train_numeric, y_train)  # post-fitting model to all train data for feature importance
#         else:
#             run_test(estimator, X_train_numeric, y_train_encoded, X_test_numeric, args.experiment_type)
#     elif args.experiment_type == 'linguistic':
#         if not args.test:
#             logger.info(f"{'-' * 50}\n\n\nPERFORMING EXPERIMENTS WITH [{estimator}]...\n{'-' * 50}")
#             do_kfold_scoring(estimator, X_train_ling, y_train, scaling=args.scaling)
#             logger.info("*" * 40, "\n")
#             estimator.fit(X_train_ling, y_train_encoded)
#         else:
#             run_test(estimator, X_train_ling, y_train_encoded, X_test_ling, args.experiment_type)
#     elif args.experiment_type == 'combined':
#         if not args.test:
#             logger.info(f"{'-' * 50}\n\n\nPERFORMING EXPERIMENTS WITH [{estimator}]...\n{'-' * 50}")
#             do_kfold_scoring(estimator, X_train_combined, y_train, scaling=args.scaling)
#             logger.info("*" * 40, "\n")
#             estimator.fit(X_train_combined, y_train_encoded)
#         else:
#             run_test(estimator, X_train_combined, y_train_encoded, X_test_combined, args.experiment_type)
#     else:
#         raise ValueError("Invalid experiment type.")
#
#     # Plot feature importance
#     if args.model_type == "rf" and args.plots:
#         importances, std = get_feature_importances(estimator)
#         save_feature_importance_plots(importances, std, args.experiment_type)
