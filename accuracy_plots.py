import subprocess
import os,sys
import matplotlib.pyplot as plt
max = 0
val_accuracy = []
filename= sys.argv[1]
# grep "INFO.*eval_accuracy =" triple_logs | awk -F "loss = " '{print $2}'

grep = subprocess.Popen(
  ["grep", "INFO.*eval_accuracy =" ,filename],
  stdout=subprocess.PIPE,
)

awk = subprocess.Popen(
  ["awk" ,"-F" ,"eval_accuracy = ", "{print $2}"],
  stdin=grep.stdout,
  stdout=subprocess.PIPE,
)
awk2 = subprocess.Popen(
  ["awk" ,"-F" ,",", "{print $1}"],
  stdin=awk.stdout,
  stdout=subprocess.PIPE,
)
for line in awk2.stdout:
  val_accuracy.append(float(line.decode('utf-8').replace('\n','')))

x = list(range(1, len(val_accuracy)+1))
plt.plot([100*i for i in x], val_accuracy, label="Validation", marker=".", color='g')
plt.xlabel('Steps sampled every 100 steps')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.legend()
plt.savefig('figs/trip_val_accuracy.png')
