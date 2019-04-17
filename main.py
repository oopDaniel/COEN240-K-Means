import random
import sys
import xlrd
import numpy as np
import matplotlib
"""
Workaround to solve bug using matplotlib. See:
https://stackoverflow.com/questions/49367013/pipenv-install-matplotlib
"""
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

EPSILON = 1e-5

class KMeans:
  def __init__(self, data, labels):
    self.data = data
    self.clusterTypes = list(set(labels)) # `label` must be integer
    self.prev_cost = float('inf')
    self.initMeans()

  def initMeans(self):
    self.means = random.sample(self.data, len(self.clusterTypes))

  def start(self):
    data = np.array(self.data)
    clusters = np.zeros(len(data))
    means = np.array(self.means)
    costs = []

    while True:
      cost = 0

      # Assign each data point to the closest mean
      for i in range(len(data)):
        distances = np.linalg.norm(data[i] - means, axis=1)
        cluster = np.argmin(distances)
        clusters[i] = cluster
        # Accumulate the cost for plotting
        cost += distances[cluster]

      costs.append(cost)

      # Deep copy
      prev_means = np.copy(means)

      # Recalculate the cluster center
      for c in self.clusterTypes:
        points = [data[idx] for idx in range(len(data)) if clusters[idx] == c]
        means[int(c)] = np.mean(points, axis=0)

      delta = np.linalg.norm(prev_means - means)
      if delta <= EPSILON:
        break

    return costs


def parse_raw_data_from_file(path):
    xlsx = xlrd.open_workbook(path)
    sheet = xlsx.sheet_by_index(0)

    data = []
    labels = []

    # Exclude the 1st row since we only need raw data
    rows = sheet.nrows - 1
    # Partition the raw data into data and labels
    for row_idx in range(rows):
      row = sheet.row_values(row_idx + 1)
      data.append(row[1:-1])
      labels.append(row[-1])

    return data, labels

def get_costs(data, labels):
  k_means = KMeans(data, labels)
  return k_means.start()

"""
Render the line chart. `matplotlib` conflicts with pipenv 😞. Check:
https://matplotlib.org/faq/osx_framework.html
"""
def plot_costs(costs):
  plt.plot(range(len(costs)), costs, 'r-o')
  plt.axis([0, len(costs), 0, max(costs) * 1.5])
  plt.xlabel('Iteration')
  plt.ylabel('J for assignment step')
  plt.show()

if __name__ == '__main__':
    file_path = sys.argv[1]

    # Parse data
    data, labels = parse_raw_data_from_file(file_path)

    # Get all costs before converge
    costs = get_costs(data, labels)

    print("Costs:", costs)

    # # Plot the accuracy with nice line chart
    plot_costs(costs)