import pandas as pd
from sklearn.cluster import KMeans

def cluster():
    model = KMeans(2)
    df = pd.read_csv("Data/congress.csv")
    print(df.head())
    model = model.fit(df[["vote1", "vote2", "vote3", "vote4", "vote5", "vote6", "vote7", "vote8", "vote9", "vote10",
                          "vote11", "vote12", "vote13", "vote14", "vote15"]])
    df["cluster"] = model.predict(df[["vote1", "vote2", "vote3", "vote4", "vote5", "vote6", "vote7", "vote8", "vote9",
                                      "vote10", "vote11", "vote12", "vote13", "vote14", "vote15"]])
    print(df[["name","party","cluster"]])
    df[["name", "party", "cluster"]]
    print(df[df.party=="D"])
    print(df[(df.party=="D") & (df.cluster==0)])