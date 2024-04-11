import llm
import time
import pandas as pd

results = []

df = pd.read_excel("android_benchmark.xlsx")
data_list = df["query"].tolist()

for query in data_list:
  start_time = time.time()
  llm.inference(query)
  results.append(time.time() - start_time)
  
df = pd.DataFrame(results)

print("Benchmark results:")
print(df)
print("\nStatistics:")
print(df.describe())