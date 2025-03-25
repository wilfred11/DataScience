from scipy.stats import expon
import matplotlib.pyplot as plt
import seaborn as sns

def exponential_info():
    print("An exponential distribution is another commonly used continuous probability distribution")
    print("which allows us to model the amount of time between two consecutive events.")
    print("It is an extension of Poisson distribution. Suppose we are using Poisson to model the number of messages in a given time period.")
    print("What if we wanted to understand the time interval between messages?")
    print("This is where the exponential distribution comes in, allowing us to model the time between each message.")
    print("We know from the Poisson distribution that lambda (λ) is the average rate at which an event occurs(number of events per unit time).")
    print("From this, it follows that the average time between 2 events is (1 / λ), which is the mean of exponential distribution.")
    print("For example, if the mean rate of messages per hour, λ, is 240,")
    print("then the average time between 2 messages would be (1/240) hrs = (3600/240) seconds = 15 seconds.")

def exponential():
    print("Plot exponential distributions given that the average time between two successive messages is 50, 60 and 70 seconds.")
    # When average time between 2 messages is 50 seconds
    data1 = expon.rvs(scale=50, size=10000)

    # When average time between 2 messages is 60 seconds
    data2 = expon.rvs(scale=60, size=10000)

    # When average time between 2 messages is 80 seconds
    data3 = expon.rvs(scale=80, size=10000)

    # Plot sample data
    sns.kdeplot(x=data1, fill=True, label='1/lambda=50')
    sns.kdeplot(x=data2, fill=True, label='1/lambda=60')
    sns.kdeplot(x=data3, fill=True, label='1/lambda=80')
    plt.xlabel('Units of time between successive events')
    plt.ylabel('Probability')
    plt.title('Exponential Distribution')
    plt.legend()
    plt.xlim(0, 200)
    plt.show()

def exponential_banner():
    print("Suppose the average number of minutes between clicks on a banner advertisement is 8 minutes.")
    print("What is the probability that  we’ll have to wait less than 5 minutes for a click?")

    # calculate probability that x is <= 5 when mean is 8
    ans = expon.cdf(x=5, scale=8)
    print('Probability that the wait time is <= five minutes:', round(ans * 100, 2))


