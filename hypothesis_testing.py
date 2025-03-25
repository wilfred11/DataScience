import random

import pandas as pd
from statsmodels.stats.weightstats import ztest

def test_slightly_unfair_die(n):
    data=[]

    for i in range(n):
        roll = random.choice([1,2,3,4,5,6,6])
        #d={"roll":roll}
        data.append(roll)
    #df=pd.DataFrame(data)
    print(data)
    expected_val_for_fair_die = (1+2+3+4+5+6)/6
    z_score,p_value = ztest(data, value=expected_val_for_fair_die)
    #print(vs)
    print("p_value: "+ str(p_value))
    print("z_score: "+ str(z_score))
    if p_value>0.05:
        print("fair die")
    else:
        print("unfair die")



