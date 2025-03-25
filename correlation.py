import math

import scipy
from scipy.stats import linregress
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def correlation():
    df=pd.read_csv("Data/diamonds.csv")
    df.plot.scatter(x="carat", y="price")
    plt.show()
    df_num = df.select_dtypes(include=np.number)
    print(df_num.corr())

def z_score(x, m, std):
    z=(x-m)/std
    return z

def correlation_lr():
    print("calculating corr for:")
    res_quiz1 = np.array([10,9,5,4])
    print("results quiz1")
    print(res_quiz1)
    res_quiz2 = np.array([10,7,9,6])
    print("results quiz2")
    print(res_quiz2)
    avg_q1 = np.mean(res_quiz1)
    print("avg")
    print(avg_q1)
    avg_q2 = np.mean(res_quiz2)
    print("avg")
    print(avg_q2)
    print("standard deviation of res_quiz1 (np)")
    std_q1 = np.std(res_quiz1, ddof=1)
    print(std_q1)
    print("standard deviation of res_quiz2 (np)")
    std_q2 = np.std(res_quiz2, ddof=1)
    print(std_q2)

    var_q1=[]
    var_q2=[]
    err_q1=[]
    err_q2=[]
    cov =[]

    for x in res_quiz1:
        var_q1.append(pow((x - avg_q1),2))
    for x in res_quiz2:
        var_q2.append(pow((x - avg_q2),2))
    for x in res_quiz1:
        err_q1.append(x - avg_q1)
    for x in res_quiz2:
        err_q2.append(x - avg_q2)

    print("var_q1 own calc")
    print(sum(var_q1))
    errs = zip(err_q1,err_q2)
    for err_1, err_2 in errs:
          cov.append(err_1*err_2)
    print(cov)
    corr=sum(cov)/math.sqrt(sum(var_q1)*sum(var_q2))
    print("corr own calc")
    print(corr)

    slope_of_regression_line = corr * (std_q2 / std_q1)
    intercept = avg_q2 - (slope_of_regression_line * avg_q1)
    print("regresion line (own calc)")
    print("y="+ str(intercept) + "+" + str(slope_of_regression_line)+"x")

    print("corrcoef")
    print(np.corrcoef(res_quiz1, res_quiz2))
    print(corr)
    plt.scatter(res_quiz1, res_quiz2)
    plt.plot(res_quiz1, intercept + slope_of_regression_line * res_quiz1, label='Linear Fit', color='red')
    plt.show()

    coefficients = np.polyfit(res_quiz1, res_quiz2, 1)
    print("Linear Fit Coefficients:", coefficients)
    p = np.poly1d(coefficients)
    plt.scatter(res_quiz1, res_quiz2, label='Data Points')
    plt.plot(res_quiz1, p(res_quiz1), label='Linear Fit', color='red')
    plt.legend()
    plt.show()

    sc=scipy.stats.linregress(res_quiz1, res_quiz2)
    (slp, inter, rvalue, pvalue, stderr) = linregress(res_quiz1, res_quiz2)
    plt.scatter(res_quiz1, res_quiz2, color="red", marker="o", label="Original data")
    y_pred = inter + slp * res_quiz1
    plt.plot(res_quiz1, y_pred, color="green", label="Fitted line")
    plt.legend(loc='best')
    plt.xlabel('res_q1')
    plt.ylabel('res_q2')
    plt.show()






