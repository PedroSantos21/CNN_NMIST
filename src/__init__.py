import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import uniform, choice

num_digits = 3

def individual(parametersRange):
  return [round(uniform(*parameter), num_digits) if type(parameter) == tuple else choice(parameter)
          for parameter in parametersRange]

def plot(hist, op="show", name=None, lang='en'):
  x = [i+1 for i in range(len(hist))]
  y = hist
  # English
  plt.plot(x, y)
  plt.xlabel("# of iterations" if lang == 'en' else "Número de iterações")
  plt.ylabel("Loss")
  if op == "show":
    plt.show()
  else:
    plt.savefig(name)
  plt.gcf().clear()