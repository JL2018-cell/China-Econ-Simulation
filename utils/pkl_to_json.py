import json
import pandas

x = pandas.read_pickle(input("Pickle file?"))

with open("data.json", "w") as f:
  json.dump(x, f)
print("saved as data.json.")
