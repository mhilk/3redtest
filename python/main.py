import abc
import os.path
import sys

import numpy as np
import pandas as pd


class ColumnProvider(abc.ABC):
    @abc.abstractmethod
    def cols(self):
        pass

    @abc.abstractmethod
    def valid_range(self):
        pass


class FeatureExtractor(ColumnProvider, abc.ABC):
    @abc.abstractmethod
    def extract(self, raw_df):
        pass


class TargetExtractor(ColumnProvider, abc.ABC):
    @abc.abstractmethod
    def extract(self, raw_df):
        pass


class ColumnSelectingExtractor(FeatureExtractor, TargetExtractor):

    def __init__(self, column_names):
        self.column_names = column_names

    def extract(self, raw_df):
        return raw_df[self.column_names]

    def cols(self):
        return self.column_names

    def valid_range(self):
        return None, None


class FeaturesV1(FeatureExtractor):
    BOOK_PRESSURE = "book_pressure"
    BOOK_PRESSURE_CHANGE_1_ROW = "book_pressure_change_1_row"
    QTY_IMBALANCE = "qty_imbalance"
    SPREAD = "spread"

    def __init__(self):
        pass

    def extract(self, raw_df):
        tob_bid_price = raw_df['bp0'].values
        tob_bid_qty = raw_df['bq0'].values
        tob_ask_price = raw_df['ap0'].values
        tob_ask_qty = raw_df['aq0'].values
        raw_df[FeaturesV1.SPREAD] = (tob_ask_price - tob_bid_price).astype(np.float64)
        raw_df.loc[(tob_bid_qty == 0) | (tob_ask_qty == 0), FeaturesV1.SPREAD] = np.inf
        bp = (tob_bid_qty.astype(np.float32) / (tob_bid_qty + tob_ask_qty))
        raw_df[FeaturesV1.BOOK_PRESSURE] = bp
        raw_df[FeaturesV1.BOOK_PRESSURE_CHANGE_1_ROW] = 0.0
        raw_df.iloc[1:, raw_df.columns.get_loc(FeaturesV1.BOOK_PRESSURE_CHANGE_1_ROW)] = (bp[1:] - bp[:-1])
        raw_df[FeaturesV1.QTY_IMBALANCE] = (
                (tob_bid_qty - tob_ask_qty).astype(np.float32) / np.sqrt(tob_bid_qty + tob_ask_qty))
        return raw_df

    def cols(self):
        return [FeaturesV1.BOOK_PRESSURE, FeaturesV1.QTY_IMBALANCE, FeaturesV1.BOOK_PRESSURE_CHANGE_1_ROW]

    def valid_range(self):
        return 1, None


class TargetsV1(TargetExtractor):
    WEIGHTED_MID_DIFF_5_ROWS = "weighted_mid_diff_5_rows"
    OFFSET = 5

    def __init__(self):
        self.col_names = [TargetsV1.WEIGHTED_MID_DIFF_5_ROWS]

    def extract(self, raw_df):
        tob_bid_price = raw_df['bp0'].values
        tob_bid_qty = raw_df['bq0'].values
        tob_ask_price = raw_df['ap0'].values
        tob_ask_qty = raw_df['aq0'].values
        spread = tob_ask_price - tob_bid_price
        weighted_mid = tob_bid_price + (spread * (tob_bid_qty.astype(np.float32) / (tob_bid_qty + tob_ask_qty)))
        weighted_mid_change = weighted_mid[TargetsV1.OFFSET:] - weighted_mid[:-TargetsV1.OFFSET]
        raw_df[TargetsV1.WEIGHTED_MID_DIFF_5_ROWS] = np.nan
        raw_df.iloc[:-TargetsV1.OFFSET,
        raw_df.columns.get_loc(TargetsV1.WEIGHTED_MID_DIFF_5_ROWS)] = weighted_mid_change
        raw_df.loc[(tob_bid_qty == 0) | (tob_ask_qty == 0), TargetsV1.WEIGHTED_MID_DIFF_5_ROWS] = np.nan
        return raw_df

    def cols(self):
        return self.col_names

    def valid_range(self):
        return 0, -5


class Scaler(abc.ABC):
    @abc.abstractmethod
    def scale(self, features):
        pass


class NormalScaler(Scaler):

    def scale(self, features):
        return (features - features.mean(axis=0)) / features.std(axis=0)


class Model(abc.ABC):
    def predict(self, X):
        pass


class LassoModel(Model):

    def __init__(self, n_feats):
        self.n_feats = n_feats
        self.weights = np.zeros(n_feats)
        self.b = 0

    def predict(self, feats):
        return feats.dot(self.weights) + self.b

    def update(self, dw, db):
        assert np.all(np.isfinite(dw)), "nan weight updates"
        assert np.all(np.isfinite(db)), "nan intercept updates"
        self.weights -= dw
        self.b -= db


class Regressor(abc.ABC):
    def fit(self, X, Y):
        pass


class LassoRegressor(Regressor):
    def __init__(self, learning_rate, iterations, l1_penalty):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty

    def fit(self, X, Y):
        n_samples, n_feats = X.shape
        model = LassoModel(n_feats)
        for i in range(self.iterations):
            Y_pred = model.predict(X)
            # calculate gradients
            dw_raw = X.T.dot(Y - Y_pred)
            dW = (-2 * dw_raw) / n_samples + self.l1_penalty * np.sign(model.weights)
            db = -2 * np.sum(Y - Y_pred) / n_samples

            # update weights
            model.update(self.learning_rate * dW, self.learning_rate * db)

        return model


class TrainTestDateProvider(abc.ABC):
    def get_dates(self):
        pass


class SequentialTrainTestDateProvider:
    def __init__(self, train_start, train_end, test_start, test_end):
        assert train_start < train_end <= test_start < test_end
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end

    def get_dates(self):
        return range(self.train_start, self.train_end), range(self.test_start, self.test_end)


class Subsampler(abc.ABC):
    @abc.abstractmethod
    def filter(self, df):
        pass


class SubsamplerV1(Subsampler):

    def filter(self, df):
        timediff = df['timestamp'].diff(-1)
        return df.iloc[timediff.values != 0]


def main():
    input_dir = sys.argv[1]
    date_provider = SequentialTrainTestDateProvider(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
                                                    int(sys.argv[5]))

    featureExtractor = FeaturesV1()
    targetExtractor = TargetsV1()
    subsampler = SubsamplerV1()

    train_dfs = []
    test_dfs = []

    train_range, test_range = date_provider.get_dates()
    for date in train_range:
        date_df = prep_date(date, featureExtractor, input_dir, targetExtractor, subsampler)
        if date_df is not None:
            train_dfs.append(date_df)

    for date in test_range:
        date_df = prep_date(date, featureExtractor, input_dir, targetExtractor, subsampler)
        if date_df is not None:
            test_dfs.append(date_df)

    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)
    regression = LassoRegressor(0.01, 1000, 1)

    target_range = targetExtractor.valid_range()
    feature_range = featureExtractor.valid_range()
    starts = [valid_range[0] for valid_range in [target_range, feature_range] if valid_range[0] is not None]
    ends = [valid_range[1] for valid_range in [target_range, feature_range] if valid_range[1] is not None]
    start_range = max(starts) if len(starts) > 0 else None
    end_range = min(ends) if len(ends) > 0 else None
    train_df = train_df.iloc[start_range:end_range]
    target = train_df[targetExtractor.cols()].values.squeeze(axis=1)
    features = train_df[featureExtractor.cols()].values
    valid_rows = np.isfinite(target) & np.all(np.isfinite(features), axis=1)
    target = target[valid_rows]
    features = features[valid_rows]

    features = NormalScaler().scale(features)
    model = regression.fit(features, target)
    yhat = model.predict(features)
    print(((target - target.mean()) ** 2).sum(), target.std())
    error = target - yhat
    print((error ** 2).sum(), error.std())

    # print(df.iloc[1000:1010])
    print('here')
    print(model.weights)

    # plt.scatter(features, Y_test, color='blue', label='Actual Data')
    # plt.plot(X_test, Y_pred, color='orange', label='Lasso Regression Line')
    # plt.title('Salary vs Experience (Lasso Regression)')
    # plt.xlabel('Years of Experience (Standardized)')
    # plt.ylabel('Salary')
    # plt.legend()
    # plt.show()


def prep_date(date, featureExtractor, input_dir, targetExtractor, subsampler):
    df = get_date_df(date, input_dir)
    if df is not None:
        df = subsampler.filter(df)
        df = featureExtractor.extract(df)
        df = targetExtractor.extract(df)
    return df


def get_date_df(date, input_dir):
    path = os.path.join(input_dir, str(date) + ".csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['date'] = date
        return df
    return None


def widen_pandas():
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


if __name__ == '__main__':
    main()
