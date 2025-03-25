import scipy.stats as S


def poisson():
    print("A hospital gets 3 patients in the ICU per day on average.")
    print("Find the probability that there will be 5 patients coming in tomorrow.")
    print("Ans: The probability of number of patients coming in per day can be represented using a Poisson Distribution with an average rate (lambda) of 3.")
    print("To calculate P(X=5), we can use the poisson.pmf() method from scipy.stats")
    # The expected value, Lambda, is the daily average rate
    mu = 3
    D=S.poisson(3)
    # Calculate probability of number of patients being 5 tomorrow
    p_5 = D.pmf(5)
    print('Probability of 5 patients coming in tomorrow:'+ str(round(p_5 * 100, 2)) + '%')
    print("the probability that there will be more than 4 patients coming in tomorrow.")
    print("Ans: Using poisson.cdf() , we can calculate the cumulative probability of 4")
    print("or less patients. Subtract the result from 1 to get the inverse probability.")
    # Calculate probability of number of patients being >4 tomorrow
    greater_4 = 1 - D.cdf(4)
    print('Probability of more than 4 patients:'+ str(round(greater_4 * 100, 2))+ "%")






