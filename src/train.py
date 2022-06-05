import argparse
from typing import Union

import numpy as np
import pandas as pd
from simpletransformers.config.model_args import ClassificationArgs
from sklearn.model_selection import StratifiedKFold

import utils_pipeline
from enums import Split, ConfigMode, Modality
from src.model import ClassificationTransformer, NeuralNetwork


def train_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # |-----------------------------------------|
    # |            BASE ARGUMENTS               |
    # |-----------------------------------------|
    parser.add_argument("-r", "--result_path", default='./transformer_predictions.txt',
                        help="Directory where to save results.")
    parser.add_argument("-tr", "--train_file", default='./data/IK_NLP_22_PESTYLE/train.tsv',
                        type=str, help="Location of training file.")
    parser.add_argument("-f", "--features", help="Which features to include", choices=["log", "ling", "both"])
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to enable or disable logging.')
    # |-----------------------------------------|
    # |            BERT MODEL                   |
    # |-----------------------------------------|
    bert_parser = subparsers.add_parser("BERT")
    bert_parser.add_argument('--subparser_bert', default=True)
    bert_parser.add_argument("-c", "--bert_config", type=str, default='dbmdz/bert-base-italian-uncased',
                             help="Specify a model configuration from HuggingFace.")
    # |-----------------------------------------|
    # |            NEURAL NETWORK               |
    # |-----------------------------------------|
    # Specified parameters based training
    nn_parser = subparsers.add_parser("NN")
    nn_parser.add_argument('-e', '--early_stopping', type=int, help="Patience of early stopping.", default=100)
    nn_parser.add_argument('-l', '--learning_rate', type=float, default=1e-3)
    nn_parser.add_argument('-b', '--batch_size', type=int, default=16)
    nn_parser.add_argument('-d', '--dropout', type=float, default=0.1)
    nn_parser.add_argument('-c', '--clipvalue', type=float, default=0.5)
    nn_parser.add_argument('-o', '--optimizer', type=str, default="Adam")
    # Parameter search training
    nn_paramsearch_parser = subparsers.add_parser("NN_ParamSearch")
    nn_paramsearch_group = nn_paramsearch_parser.add_mutually_exclusive_group()
    # TODO: Define values for GridSearchCV later
    nn_paramsearch_group.add_argument('--grid', action='store_true',
                                      help="Perform parameter search with 'GridSearchCV'.")
    # TODO: Define distribution for RandomizedSearchCV
    nn_paramsearch_group.add_argument('--random', action='store_true',
                                      help="Perform parameter search with 'RandomizedSearchCV'.")

    # |-----------------------------------------|
    # |            REGRESSION ML MODELS         |
    # |-----------------------------------------|
    regressor_parser = subparsers.add_parser("ML_REG")
    # Common regressor params
    regressor_parser.add_argument('-m', '--model', type=str, default=None, required=True, choices=['lr', 'rr', 'en'],
                                  help="'lr': LinearRegression, 'rr': Ridge, 'en': ElasticNet")
    regressor_parser.add_argument('-n', '--normalize', action='store_true', help="Subtract mean and divide by L2-norm.")
    regressor_parser.add_argument('--n_jobs', type=int, default=-1)
    # Ridge() and ElasticNet() params
    regressor_parser.add_argument('-a', '--alpha', type=Union[float, int], default=1,
                                  help="Control L2 term by multiplication with alpha, in [0, inf].")
    regressor_parser.add_argument('-s', '--solver', type=str, default='auto',
                                  choices=['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'])
    # ElasticNet() params
    # TODO: always add 'precompute'
    regressor_parser.add_argument('-l1', '--l1_ratio', type=float, default=0.5,
                                  help='For l1=0 the penalty is an L2 penalty. For l1=1 it is an L1 penalty.')
    regressor_parser.add_argument('--max_iter', type=int, default=1000)
    regressor_parser.add_argument('--tol', type=float, default=1e-4, help='Tolerance for optimization.')
    regressor_parser.add_argument('--selection', type=str, default='random',
                                  help='Setting ‘random’ often leads to significantly faster convergence,'
                                       ' especially when tol is higher than 1e-4.')

    regressor_paramsearch_parser = subparsers.add_parser("ML_REG_ParamSearch")
    # TODO: Define values for GridSearchCV later
    regressor_paramsearch_parser.add_argument('--grid', action='store_true',
                                              help="Perform parameter search with 'GridSearchCV'.")
    # TODO: Define distribution for RandomizedSearchCV
    regressor_paramsearch_parser.add_argument('--random', action='store_true',
                                              help="Perform parameter search with 'RandomizedSearchCV'.")
    # |-----------------------------------------|
    # |           CLASSIFIER ML MODELS          |
    # |-----------------------------------------|
    # 1) LinearSVC
    svc_parser = subparsers.add_parser("ML_CLF_SVC")
    svc_parser.add_argument('--penalty', type=str, choices=["l1", "l2"])
    svc_parser.add_argument('--loss', type=str, choices=["hinge", "squared_hinge"])
    svc_parser.add_argument('--C', type=float, help="Inversely proportional regularization parameter.", default=1.0)
    svc_parser.add_argument('--max_iter', type=int, default=1000)
    svc_paramsearch_parser = subparsers.add_parser("ML_CLF_SVC_ParamSearch")
    # TODO: Define values for GridSearchCV later
    svc_paramsearch_parser.add_argument('--grid', action='store_true',
                                        help="Perform parameter search with 'GridSearchCV'.")
    # TODO: Define distribution for RandomizedSearchCV
    svc_paramsearch_parser.add_argument('--random', action='store_true',
                                        help="Perform parameter search with 'RandomizedSearchCV'.")
    # 2) RandomForest
    randomforest_parser = subparsers.add_parser("ML_CLF_RF")
    randomforest_parser.add_argument('-ne', type=int, default=100, help='Number of estimators in the forest.', )
    randomforest_parser.add_argument('--max_depth', type=int, default=None,
                                     help="None means expansion until all leaves are pure,"
                                          " or contain less than min_samples")
    # Modifying this to higher than 2 is an immediate regularization
    randomforest_parser.add_argument('--min_samples', type=int, default=2)
    randomforest_paramsearch_parser = subparsers.add_parser("ML_CLF_RF_ParamSearch")
    # TODO: Define values for GridSearchCV later
    randomforest_paramsearch_parser.add_argument('--grid', action='store_true',
                                                 help="Perform parameter search with 'GridSearchCV'.")
    # TODO: Define distribution for RandomizedSearchCV
    randomforest_paramsearch_parser.add_argument('--random', action='store_true',
                                                 help="Perform parameter search with 'RandomizedSearchCV'.")

    return parser.parse_args()


if __name__ == "__main__":
    args = train_parser()

    # Create the correct splits for cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    df_train = utils_pipeline.load_dataframe(split=Split.TRAIN, mode=ConfigMode.FULL,
                                             exclude_modality=Modality.SCRATCH, only_numeric=True)

    if hasattr(args, 'subparser_bert'):
        # Define model
        model_args = ClassificationArgs()
        model_args.num_train_epochs = 3
        model_args.overwrite_output_dir = True
        model_args.learning_rate = 0.000001
        model_args.do_lower_case = True
        model_args.silence = True
        model = ClassificationTransformer(args, model_args)

        X = None
        y = None
    else:
        if hasattr(args, 'subparser_nn'):
            model = NeuralNetwork(input_len, layers=layers, LR=LR)
        elif hasattr(args, 'subparser_nn_search'):
            pass
        elif hasattr(args, 'subparser_ml_reg'):
            pass
        elif hasattr(args, 'subparser_ml_reg_search'):
            pass
        elif hasattr(args, 'subparser_ml_clf'):
            pass
        elif hasattr(args, 'subparser_ml_clf_search'):
            pass
        else:
            raise SyntaxError("In order to perform training, specification of subparser is required.")

        # if args.features == "log", no change is required
        if args.features == 'ling':
            df_train = utils_pipeline.get_ling_feats(Split.TRAIN)
        elif args.features == 'both':
            df_train_ling = utils_pipeline.get_ling_feats()
            df_train = pd.concat([df_train, df_train_ling], axis='columns')
        X = df_train.to_numpy()
        y = utils_pipeline.get_train_labels()
        y_encoder = utils_pipeline.get_train_label_encoder()

        # Do StratifiedKfold training
        for train_index, valid_index in skf.split(X, y):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
            y_train_encoded, y_valid_encoded = y_encoder.transform(y[train_index]), y_encoder.transform(y[valid_index])
            X_train_scaled, X_valid_scaled = utils_pipeline.scaling(X_train, X_valid)

            model.fit(X_train_scaled, y_train_encoded)
            prediction = model.predict(X_valid_scaled)


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

    X_train_numeric = select_columns(df_train, dtype_include=['float32', 'int32'], name_exclude=['item_id'])
    X_test_numeric = select_columns(df_test, dtype_include=['float32', 'int32'], name_exclude=['item_id'])

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
