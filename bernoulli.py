import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import bernoulli
from scipy.stats import normaltest

def central_limit_theorema_simul(k):
    print("generating wins from a Roulette wheel when choosing Red")
    print("There is a chance of rolling Red: 18/38")
    D =bernoulli(p=18/38)
    v=D.rvs()
    if v==1:
        print("win")
    else:
        print("loose")

    data = []
    for i in range(10000):
        s=D.rvs(k).sum()
        d={"s": s}
        data.append(d)
    df=pd.DataFrame(data)
    df.plot.hist(bins=15)
    plt.show()
    stat, p = normaltest(df)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
