
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom
import scipy.stats
import pandas as pd


def mean_variance(n,p):
    print("chance of success p=" + str(p))
    print("chance of failure q=" + str(1-p))
    print("number of tries n="+str(n))
    mean, variance = binom.stats(n,p)
    print("mean(n*p):"+str(mean))
    print("variance(n*p*q):"+str(variance))

"""
Generate a series of Head and Tails (n) using a variable p value
"""
def create_binom_data(n,p):
    data_binom = binom.rvs(n=1, p=p, size=n)
    print(data_binom[0])
    heads = 0
    tails = 0
    count = 0
    result =""

    for coin in data_binom:
        count += 1
        if coin == 1:
            heads += 1
            result+="H"
        elif coin == 0:
            tails += 1
            result += "T"
    print("randomly generated binomial series(chance of success (Heads) p="+ str(p)+")")
    print(result)
    print("Observed " + str(count) + " of coin tossing with heads " + str(heads)
          + ", tails " + str(tails))
    return data_binom

def draw():
    random_binom_numbers = binom.rvs(n=50, p=0.5, size=100)
    print("length:"+str(len(random_binom_numbers)))
    print("length:" + str(len(random_binom_numbers[0])))
    for i in random_binom_numbers:
        print(i)

    plt.figure(figsize=(10, 5))
    plt.hist(
        random_binom_numbers,
        density=True,
        bins=int(round(len(np.unique(random_binom_numbers))/2,0)),
        color="#fc9d12",
        edgecolor="grey",
    )  # density=False would make counts
    plt.xlabel("Students passing the final exam")
    plt.ylabel("Probability")
    plt.title("Binomial Probability Distribution \nfor size=25 and p=0.3")
    plt.xticks(
        np.arange(min(random_binom_numbers), max(random_binom_numbers) + 1, 2.0)
    )  # define x-axis ticks

    plt.show()


def coin():
    """
    0 heads is 25% of the time (TT)
    1 head is 50% of the time (HT, TH)
    2 heads is 25% of the time (HH)
    heads is success
    Using our two-coin flip example where COIN = binom(n=2, p=0.5), the CDF functions are asking the following:

    """
    COIN = binom(n=2, p=0.5)
    print("what percentage of results have 0.2 or fewer heads?")
    print(COIN.cdf(0.2))
    print("what percentage of results have 1 or fewer heads?")
    print(COIN.cdf(1))
    print("what percentage of results has 2 or fewer heads?")
    print(COIN.cdf(2))
    print("what is the 20%-tile of heads?")
    print(COIN.ppf(0.2))
    print("what is the 60%-tile of heads?")
    print(COIN.ppf(0.6))
    print("what is the 99%-tile of heads?")
    print(COIN.ppf(0.99))
    print("what is the 75%-tile of heads?")
    print(COIN.ppf(0.75))
    print("what is the 76%-tile of heads?")
    print(COIN.ppf(0.76))
    print("the probability of zero heads is 25%")
    print(COIN.pmf(0))
    print("the probability of one head is 50%")
    print(COIN.pmf(1))
    print("the probability of two heads is 25%")
    print(COIN.pmf(2))
    print("The .rvs() function returns a random sample of the distribution with probability equal to the distribution -- if something is 80% likely, that value will be sampled 80% of the time. In COIN, we expect more results with 1 (50% occurrence of 1 head) than 0 or 2 (25% occurrence of either zero heads or two heads).")
    c=COIN.rvs(500)
    df = pd.DataFrame(c)
    print(df.value_counts())