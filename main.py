import pandas as pd
from numpy.random import random
from scipy import stats
from scipy.stats import skew
import numpy as np

from bernoulli import central_limit_theorema_simul
from cluster import cluster
from correlation import correlation, correlation_lr
from exponential import exponential, exponential_info, exponential_banner
from hypothesis_testing import test_slightly_unfair_die
import random
from binom import create_binom_data, mean_variance, draw, coin
from normal import normal_visual, normal_distribution, confidence_interval
from linear_regression import linear_regression, multiple_regression
from poisson import poisson

do= 11

if do==0:
    path = "Data/IBM-313 Marks.xlsx"
    table = pd.read_excel(path)
    print(table)

    x= table['Total']
    print(np.mean(x))
    print(np.median(x))

    print(stats.mode(x))

    a = np.array([1,2,300,4,4,3,3,3,3,4,20,5, 6,2])
    p= np.percentile(a, 50)
    print(p)

    print(skew(x))

    from matplotlib import pyplot as plt

    plt.boxplot(x,sym='*')
    plt.show()


    counts, bins = np.histogram(x, bins=15)



    # Creating dataset
    x = np.random.randint(100, size=50)

    # Creating plot
    fig = plt.figure(figsize=(10, 7))

    #plt.hist(a, bins=[0, 10, 20, 30,
    #                  40, 50, 60, 70,
    #                  80, 90, 100])
    plt.hist(x, bins=bins)

    plt.title("Numpy Histogram")

    # show plot
    plt.show()

if do == 1:
    n=100
    p=0.6
    mean_variance(n,p)
    data_binom = create_binom_data(n,p)
    draw()

if do == 2:
    random.seed(4)
    test_slightly_unfair_die(100)
    test_slightly_unfair_die(10000)

if do==3:
    normal_visual()
    normal_distribution()

if do==4:
    coin()

if do==5:
    k = 1
    central_limit_theorema_simul(k)
    k = 100000
    central_limit_theorema_simul(k)

if do == 6:
    confidence_interval()

if do == 7:
    correlation()
    correlation_lr()

if do == 8:
    linear_regression()
    multiple_regression()

if do == 9:
    cluster()

if do == 10:
    poisson()

if do == 11:
    exponential_info()
    exponential()
    exponential_banner()


