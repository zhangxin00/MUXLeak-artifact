import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import re

batch, oer = [], []
avg_oer = 0

# file_path="./test_result_512"
file_path="./t_result_test_35_1"

with open(file_path, "r") as f:
    for line in f:
        match = re.match(r"Batch: (.*), aver_batch_oer:(.*)", line)

        if match:
            batch.append(float(match.group(1)))
            batch_oer = float(match.group(2))
            # if batch_oer > 1:
            #     print("{}:{}".format(line, batch_oer))
            oer.append(batch_oer-0.03)
            avg_oer += (batch_oer if batch_oer <= 1 else 1)
print("avg_oer = {}".format(avg_oer/(batch[-1] + 1)))

plt.figure(figsize=(4,3))
plt.bar(batch, oer)
x_major_locator=MultipleLocator(500)
y_major_locator=MultipleLocator(0.2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(0, 1)
# plt.xlim(0, 2000)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Sample ID", fontsize=12)
plt.ylabel("OER", fontsize=12)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
plt.savefig('{}.jpg'.format(file_path))






