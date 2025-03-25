import pandas as pd
from sklearn.linear_model import LinearRegression


def linear_regression():
    model = LinearRegression()
    df = pd.read_csv("Data/diamonds.csv")
    model = model.fit(df[["carat"]], df["price"])
    data = []
    data.append({"carat": 1})
    data.append({"carat": 2})
    data.append({"carat": 3})
    df2 = pd.DataFrame(data)
    df2["price_predict"] = model.predict(df2)
    print(df2)

def multiple_regression():
    simple_model = LinearRegression()
    model = LinearRegression()
    df = pd.read_csv("Data/diamonds.csv")
    model.fit(df[["carat","table"]], df["price"])
    simple_model = simple_model.fit(df[["carat"]], df["price"])
    df["price_simple"]= simple_model.predict(df[["carat"]])
    df["price_multi"] = model.predict(df[["carat","table"]])
    df["simple_error"]= df["price"] - df["price_simple"]
    df["multi_error"] = df["price"] - df["price_multi"]
    df["abs_err_simple"] =abs(df["simple_error"])
    df["abs_err_multi"] = abs(df["multi_error"])
    print("multi_err_mean: "+ str(df["abs_err_multi"].mean()))
    print("simple_err_mean: " + str(df["abs_err_simple"].mean()))
    err_imp= df["abs_err_multi"].mean() - df["abs_err_simple"].mean()
    print("multiple regression using carat and table only lessens the error by "+ str(err_imp))

