import subprocess
import os,sys
import matplotlib.pyplot as plt
max = 0
training_loss = []
val_loss = []
filename= sys.argv[1]
# grep "INFO.*eval_accuracy =" triple_logs | awk -F "loss = " '{print $2}'

grep = subprocess.Popen(
  ["grep", "INFO.*eval_accuracy =" ,filename],
  stdout=subprocess.PIPE,
)

awk = subprocess.Popen(
  ["awk" ,"-F" ,"loss = ", "{print $2}"],
  stdin=grep.stdout,
  stdout=subprocess.PIPE,
)
for line in awk.stdout:
  val_loss.append(float(line.decode('utf-8').replace('\n','')))

grep = subprocess.Popen(
  ["grep", "] loss =" ,filename],
  stdout=subprocess.PIPE,
)

awk = subprocess.Popen(
  ["awk" ,"-F" ,"loss = ", "{print $2}"],
  stdin=grep.stdout,
  stdout=subprocess.PIPE,
)

awk2 = subprocess.Popen(
  ["awk" ,"-F" ,",", "{print $1}"],
  stdin=awk.stdout,
  stdout=subprocess.PIPE,
)
for line in awk2.stdout:
  training_loss.append(float(line.decode('utf-8').replace('\n','')))

max = len(training_loss)
if(len(val_loss) > max):
    max = len(val_loss)
l = training_loss[len(training_loss)-1]
while(len(training_loss) < max):
    training_loss.append(l)

l = val_loss[len(val_loss)-1]
while(len(val_loss) < max):
    val_loss.append(l)
x = list(range(1, len(training_loss)+1))
plt.plot([100*i for i in x], val_loss, label="Validation Loss", marker=".", color='r')
plt.plot([100*i for i in x], training_loss, label="Training Loss", marker=".", color='g')
# plt.plot([50*i for i in x], Task3, label="Reordering", marker=".", color='g')
# plt.plot([50*i for i in x], Task4, label="Deletion", marker=".", color='b')
plt.xlabel('Steps sampled every 100 steps')
# plt.ylabel('Tr')
plt.tight_layout()
plt.legend()
plt.savefig('figs/trip_loss.png')
