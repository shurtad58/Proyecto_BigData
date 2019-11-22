from pyspark import SparkContext
from random import randint
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.feature import HashingTF, IDF

sc = SparkContext(appName="kmeans")

docs = sc.wholeTextFiles("hdfs:./articles1.csv")
words = docs.values().map(lambda line: line.split(" "))

hashingTF = HashingTF()
tf = hashingTF.transform(words)
idf = IDF().fit(tf)
tfidf = idf.transform(tf)

k = randint(3,6)

clusters = KMeans.train(tfidf, k, maxIterations=10, initializationMode="random")

print(clusters.predict(tfidf).collect())
print("")

docsArray = docs.keys().collect()
centroidsArray = clusters.predict(tfidf).collect()
for x in range(k):
    print("en el cluster " + str(x) +  " estan:")
    for j in range(centroidsArray.__len__()):
        if centroidsArray[j]==x:
            print(docsArray[j])
    print("")

