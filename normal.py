import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import norm
from numpy import random
from numpy import random
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def normal_visual():
    rng = np.random.default_rng()
    numbers = rng.normal(size=10_000)
    print("mean: "+ str(numbers.mean()))
    print("standard dev.: " + str(numbers.std()))
    sns.histplot(numbers, kde=True)
    plt.show()

def normal_distribution():
    D= norm()
    print("area left of 2:")
    print(D.cdf(2))
    print("area to the left of 0:")
    print(D.cdf(0))
    print("50% of the area is left of which z-score")
    print(D.ppf(.5))
    print("to be a one-percenter, I need more than this Z-score")
    print(D.ppf(.99))
    print("check extremes")
    print(D.ppf(1.5))
    print("what percentage of the data is between Z-scores 1 and 2?")
    print(D.cdf(2)-D.cdf(1))

def confidence_interval():
    print("Our sample average or sample percent can be used as estimates for the population average or population percent. But how confident are we that those estimates are correct?")
    print("Interpretation of CI")
    print("I am 95% sure that the true population mean lies within my confidence interval.")
    print("If hundreds of people took samples of size n, calculated the mean of their samples, and computed 95% confidence intervals, 95% of those intervals would contain the true population mean.")

    # Example data: exam scores
    #exam_scores = [75, 80, 85, 90, 92, 78, 88, 95, 79, 82]
    D= scipy.stats.norm(60,15)
    exam_scores=D.rvs(10)
    #rng = np.random.default_rng()
    #exam_scores = rng.normal(low=0, high=100,size=50)

    # Calculating sample mean
    sample_mean = np.mean(exam_scores)

    # Calculating standard error
    standard_error = np.std(exam_scores) / np.sqrt(len(exam_scores))

    # Setting confidence level
    confidence_level = 0.95

    # Calculating Z-score
    z_score = norm.ppf((1 + confidence_level) / 2)

    # Calculating confidence interval
    confidence_interval = (
        sample_mean - z_score * standard_error,
        sample_mean + z_score * standard_error
    )

    print(f"Sample Mean: {sample_mean}")
    print(f"Standard Error: {standard_error}")
    print(f"Z-Score: {z_score}")
    print(f"Confidence Interval: {confidence_interval}")

    # Plotting the data
    plt.hist(exam_scores, bins=10, color='skyblue', alpha=0.7, label='Exam Scores')

    # Plotting the confidence interval
    plt.axvline(confidence_interval[0], color='red', linestyle='dashed', linewidth=2, label='Confidence Interval')
    plt.axvline(confidence_interval[1], color='red', linestyle='dashed', linewidth=2)

    plt.xlabel('Exam Scores')
    plt.ylabel('Frequency')
    plt.title('Normal Distribution Confidence Interval')
    plt.legend()
    plt.show()

def confidence_interval():
    df = pd.read_csv("Data/gpa.csv")
    sample = df.sample(10000)
    mean = sample["Students"].mean()
    std = sample["Students"].std()
    print("mean:"+str(mean))
    print("std:"+ str(std))
    D=norm(mean, std)
    low,hi=D.interval(.68)
    print("We are 68% confident that the true number of students in an average illinois course is between "+ str(low)+" and "+str(hi))


def t_test():
    a = [rd.gauss(48, 20) for x in range(30)]
    b = [rd.gauss(56, 15) for x in range(30)]

    # Independent sample test
    # Null hypothesis: mean of a = mean of b
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    print("T-statistic:", t_stat)
    print("P-value:", p_value)
    print("mean(a), mean(b)")
    print(np.mean(a), np.mean(b))
    if p_value>0.05:
        print("P_value is greater than 0.05 so there are no reasons to reject the null hypothesis. The null hypothesis is that the means are equal.")
    else:
        print("P_value is smaller than 0.05 so there are reasons to reject the null hypothesis. The alternative hypothesis is that the means are different.")



