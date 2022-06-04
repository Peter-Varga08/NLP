import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils_pipeline
from enums import Split

if __name__ == "__main__":

    #TODO: TRANSFORMER
    # ------------------------------------------------------------------------------------------------------------------
    # Define model
    model_args = ClassificationArgs()
    model_args.num_train_epochs = 3
    model_args.overwrite_output_dir = True
    model_args.learning_rate = 0.000001
    model_args.do_lower_case = True
    model_args.silence = True
    model = ClassificationModel("bert", "dbmdz/bert-base-italian-uncased", args=model_args, use_cuda=False,
                                num_labels=3)

    # Create the correct splits for cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    X, y = prepare_train(args.train_file)
    X = np.array(X)
    y = np.array(y)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Prepare train set
        train_data = reformat_train(X_train, y_train)
        train_df = pd.DataFrame(train_data)
        train_df.columns = ["text_a", "text_b", "labels"]
        train_df["labels"] = train_df["labels"].replace(["t1", "t2", 't3'], [0, 1, 2])
        print(train_df)

        # Train model
        model.train_model(train_df)

        # Predict synsets
        predictions = predict_test(reformat_test(X_test), model)
        all_predictions.append(predictions)
        all_predictions.append("\n")
        print("ACCURACY: ", get_accuracy(y_test, predictions))

    #TODO: Neural Network
    # ------------------------------------------------------------------------------------------------------------------

    print('Hello World! This script is to train the neural network :)')
    result_path, features, oversampling, kfold = utils.parse_nn_model()
    df_train = utils_pipeline.load_dataframe(split=Split.TRAIN,
                                             mode='full',
                                             exclude_modality='ht',
                                             only_numeric=True)
    y, label_encoder = utils_pipeline.get_train_labels()
    print(features)
    if features == 'ling':
        df_train, _ = utils_pipeline.get_ling_feats()
        if kfold == 'L':
            ths = [
                0, .5, .1, .15, .2, .25, .3, 0.35, .4, 0.45, .5, .55, .6, .65,
                .7, .75, .8, .85, .9, .95
            ]
            losses = []
            accs = []
            for th in ths:
                print(th)
                df2 = utils_pipeline.filter_features(df_train, th=th, verbose=False)
                columns = list(df2.columns)
                input_len = len(columns)
                print(input_len)
                if input_len == 0:
                    break
                abc = do_single(df2,
                                y,
                                label_encoder,
                                input_len,
                                patience=100,
                                LR=1e-3,
                                oversampling=oversampling,
                                result_path=result_path)
                losses.append(abc[0])
                accs.append(abc[1])
            plt.figure(figsize=(15, 15))
            plt.plot(losses)
            plt.savefig(result_path + '/ling_loss.png')
            plt.figure(figsize=(15, 15))
            plt.plot(accs)
            plt.savefig(result_path + '/ling_acc.png')
            df_train = utils_pipeline.filter_features(df_train,
                                                      th=ths[np.argmax(accs)],
                                                      verbose=False)
        else:
            print(kfold)
    elif features == 'both':
        df_train2, _ = utils_pipeline.get_ling_feats()
        print(df_train2.shape)
        df_train2 = utils_pipeline.filter_features(df_train2, th=0, verbose=False)
        print(df_train2.shape)
        df_train2.index = df_train.index
        df_train = pd.concat([df_train, df_train2], axis='columns')
        print(df_train.shape)
        print(df_train)
        # set_trace()
    columns = list(df_train.columns)
    input_len = len(columns)
    print(input_len)
    if kfold:
        do_kfold_scoring(df_train,
                         y,
                         label_encoder,
                         input_len,
                         oversampling=oversampling)
    do_single(df_train,
              y,
              label_encoder,
              input_len,
              oversampling=oversampling,
              result_path=result_path)

    # TODO: ML models
    # ------------------------------------------------------------------------------------------------------------------
    def run_numeric_data_experiments(model: Union[LinearRegression, RandomForestClassifier],
                                     X_train_numeric: pd.DataFrame, y: np.ndarray, scaling=False) -> None:
        """Run three type of experiments: [Using KBest features, using Keystroke features, using post-edit features]."""
        print(f"{'-' * 50}\n\n\nPERFORMING EXPERIMENTS WITH [{model}]...\n{'-' * 50}")
        # Train and validate on TOP 100%, 75%, 50% and 25% features of the data
        for ratio in [1, 0.75, 0.5, 0.25]:
            kbest_ = SelectKBest(chi2, k=int(len(X_train_numeric.columns) * ratio)).fit(X_train_numeric, y)
            print(f"Performing 10-Fold CV on top {int(ratio * 100)}% features of the data...")
            do_kfold_scoring(model, X_train_numeric, y, selector=kbest_, scaling=scaling)
            print("*" * 40, "\n")

        # Train and validate on keystroke features data
        print(f"\n{'*' * 40}\n[KEYSTROKE FEATURES]")
        keystroke_columns = [col for col in X_train_numeric.columns if col.startswith('k_')]
        do_kfold_scoring(model, X_train_numeric[keystroke_columns], y, scaling=scaling)

        # Train and validate on postedit features data
        print(f"\n{'*' * 40}\n[POSTEDIT FEATURES]")
        postedit_columns = [col for col in X_train_numeric.columns if col.startswith('n_')]
        do_kfold_scoring(model, X_train_numeric[postedit_columns], y, scaling=scaling)


    # |-------------------------|
    # |     Load datasets       |
    # |-------------------------|
    logger.info("Loading datasets...")
    df_train: pd.DataFrame = load_dataset("GroNLP/ik-nlp-22_pestyle", "full",
                                          data_dir='../IK_NLP_22_PESTYLE')['train'].to_pandas()
    df_test: pd.DataFrame = load_dataset("GroNLP/ik-nlp-22_pestyle", "mask_subject",
                                         data_dir='../IK_NLP_22_PESTYLE')['test'].to_pandas()
    df_train = df_train[df_train.modality != 'ht']
    df_test = df_test[df_test.modality != 'ht']
    y_train = np.array(df_train.subject_id)
    y_test = pd.read_csv('../data/test_ground_truths.csv', header=None).to_numpy()
    # one label-encoder is enough, categories are the same for both datasets
    assert np.array_equal(np.unique(y_train),
                          np.unique(y_test)), "Unique labels within train and test set are different."
    label_encoder = LabelBinarizer().fit(y_train)
    y_train_encoded = label_encoder.transform(y_train)  # used for 'run_test()' only

    # Numeric features
    logger.info("Creating numeric features.")
    X_train_numeric = select_columns(df_train, dtype_include=['float32', 'int32'], name_exclude=['item_id'])
    X_test_numeric = select_columns(df_test, dtype_include=['float32', 'int32'], name_exclude=['item_id'])
    # Linguistic features
    logger.info("Loading linguistic features...")

    lingfeat = [
        LingFeatDF(pd.read_csv('../data/linguistic_features/train_mt.csv', sep="\t").drop(columns=['Filename']), 'train_mt'),
        LingFeatDF(pd.read_csv('../data/linguistic_features/test_mt.csv', sep='\t').drop(columns=['Filename']), 'test_mt'),
        LingFeatDF(pd.read_csv('../data/linguistic_features/train_tgt.csv', sep="\t").drop(columns=['Filename']),
                   'train_tgt'),
        LingFeatDF(pd.read_csv('../data/linguistic_features/test_tgt.csv', sep="\t").drop(columns=['Filename']), 'test_tgt'),
    ]
    intersected_linguistic_columns = intersect_linguistic_columns(*[nt.df for nt in lingfeat])
    col_lens = []
    for idx, nt in enumerate(lingfeat):
        lingfeat[idx] = filter_linguistic_columns(nt.df, intersected_linguistic_columns)
        col_lens.append(lingfeat[idx].df.columns)
    assert len(set(col_lens)) == 1, "Number of columns have to be equal for all linguistic feature dataframes."

    X_train_ling = subtract_df(lingfeat[0], lingfeat[2])
    X_test_ling = subtract_df(lingfeat[1], lingfeat[3])
    # Combined features
    X_train_combined = pd.concat([X_train_numeric.reset_index(drop=True), X_train_ling.reset_index(drop=True)], axis=1)
    X_test_combined = pd.concat([X_test_numeric.reset_index(drop=True), X_test_ling.reset_index(drop=True)], axis=1)

    # |-------------------------|
    # |       Load models       |
    # |-------------------------|
    if args.model_type == 'lr':
        estimator = LinearRegression(n_jobs=-1)
    elif args.model_type == 'rf':
        estimator = RandomForestClassifier(n_jobs=-1, random_state=42)
    else:
        raise ValueError('Invalid model type.')

    # |-------------------------|
    # |     Run experiments     |
    # |-------------------------|
    if args.experiment_type == 'numeric':
        if not args.test:
            run_numeric_data_experiments(estimator, X_train_numeric, y_train, scaling=args.scaling)
            estimator.fit(X_train_numeric, y_train)  # post-fitting model to all train data for feature importance
        else:
            run_test(estimator, X_train_numeric, y_train_encoded, X_test_numeric, args.experiment_type)
    elif args.experiment_type == 'linguistic':
        if not args.test:
            logger.info(f"{'-' * 50}\n\n\nPERFORMING EXPERIMENTS WITH [{estimator}]...\n{'-' * 50}")
            do_kfold_scoring(estimator, X_train_ling, y_train, scaling=args.scaling)
            logger.info("*" * 40, "\n")
            estimator.fit(X_train_ling, y_train_encoded)
        else:
            run_test(estimator, X_train_ling, y_train_encoded, X_test_ling, args.experiment_type)
    elif args.experiment_type == 'combined':
        if not args.test:
            logger.info(f"{'-' * 50}\n\n\nPERFORMING EXPERIMENTS WITH [{estimator}]...\n{'-' * 50}")
            do_kfold_scoring(estimator, X_train_combined, y_train, scaling=args.scaling)
            logger.info("*" * 40, "\n")
            estimator.fit(X_train_combined, y_train_encoded)
        else:
            run_test(estimator, X_train_combined, y_train_encoded, X_test_combined, args.experiment_type)
    else:
        raise ValueError("Invalid experiment type.")

    # Plot feature importance
    if args.model_type == "rf" and args.plots:
        importances, std = get_feature_importances(estimator)
        save_feature_importance_plots(importances, std, args.experiment_type)
