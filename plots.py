# %%
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 18})  # For general font
plt.rcParams.update({"axes.labelsize": 18})


#######################################
# # Distilbert standard A2T
# a, b, c, d = 60.11, 58.24, 62.31, 61.30
# # Distilbert standard PSO
# a, b, c, d = 53.39, 52.67, 54.96, 53.96
# # Distilbert standard TextBugger
# a, b, c, d = 37.15, 37.15, 40.26, 39.54
# # Distilbert standard TextFooler
# a, b, c, d = 36.81, 34.84, 41.73, 38.98

# # RoBERTaBase standard A2T
# a, b, c, d = 61.78, 60.81, 64.11, 61.94
# # RoBERTaBase standard PSO
# a, b, c, d = 53.34, 52.94, 54.40, 53.35
# # RoBERTaBase standard TextBugger
# a, b, c, d = 40.46, 39.00, 43.20, 40.53
# # RoBERTaBase standard TextFooler
# a, b, c, d = 33.19, 32.10, 37.35, 33.83

# # BERT-large  standard A2T
# a, b, c, d = 75.75, 75.68, 75.54, 75.60
# # BERT-large  standard PSO
# a, b, c, d = 67.72, 67.69, 67.55, 67.60
# # BERT-large  standard TextBugger
# a, b, c, d = 64.59, 64.53, 64.41, 64.36
# # BERT-large  standard TextFooler
# a, b, c, d = 58.48, 58.25, 58.27, 58.12

# # RoBERTaLarge  standard A2T
# a, b, c, d = 65.10, 64.89, 64.95, 64.38
# # RoBERTaLarge  standard PSO
# a, b, c, d = 55.83, 55.04, 56.70, 55.31
# # RoBERTaLarge  standard TextBugger
# a, b, c, d = 44.61, 42.20, 42.43, 41.29
# # RoBERTaLarge  standard TextFooler
# a, b, c, d = 36.63, 35.52, 37.29, 35.39

# #######################################
# # Distilbert AT A2T
# a, b, c, d = 72.09, 71.12, 71.81, 71.97
# # Distilbert AT PSO
# a, b, c, d = 55.80, 54.72, 57.87, 56.35
# # Distilbert AT TextBugger
# a, b, c, d = 38.91, 38.98, 41.64, 40.98
# # Distilbert AT TextFooler
# a, b, c, d = 40.15, 38.21, 43.81, 41.12

# # RoBERTaBase AT A2T
# a, b, c, d = 76.63, 75.82, 77.08, 76.28
# # RoBERTaBase  AT PSO
# a, b, c, d = 55.49, 54.76, 56.45, 54.99
# # RoBERTaBase  AT TextBugger
# a, b, c, d = 41.71, 40.22, 43.35, 41.33
# # RoBERTaBase  AT TextFooler
# a, b, c, d = 34.48, 32.47, 39.39, 35.29

# # BERT-large AT A2T
# a, b, c, d = 86.13, 86.03, 85.76, 85.92
# # BERT-large  AT PSO
# a, b, c, d = 70.21, 70.26, 70.38, 70.25
# # BERT-large  AT TextBugger
# a, b, c, d = 69.74, 69.89, 69.55, 69.62
# # BERT-large  AT TextFooler
# a, b, c, d = 65.43, 65.25, 65.27, 65.10

# # RoBERTaLarge AT A2T
# a, b, c, d = 81.91, 81.72, 81.62, 81.80
# # RoBERTaLarge  AT PSO
# a, b, c, d = 57.99, 57.29, 59.71, 58.18
# # RoBERTaLarge  AT TextBugger
# a, b, c, d = 45.00, 43.54, 44.74, 43.53
# # RoBERTaLarge  AT TextFooler
# a, b, c, d = 39.64, 37.06, 42.44, 39.87

# #######################################
# # Distilbert FreeLB A2T
# a, b, c, d = 59.63, 57.90, 62.95, 61.88
# # Distilbert FreeLB PSO
# a, b, c, d = 54.40, 53.91, 56.86, 55.89
# # Distilbert FreeLB TextBugger
# a, b, c, d = 32.79, 32.24, 37.80, 36.21
# # Distilbert FreeLB TextFooler
# a, b, c, d = 32.49, 31.22, 39.64, 37.39

# # RoBERTaBase FreeLB A2T
# a, b, c, d = 65.93, 65.04, 68.85, 66.59
# # RoBERTaBase FreeLB PSO
# a, b, c, d = 53.64, 52.99, 55.24, 53.55
# # RoBERTaBase FreeLB TextBugger
# a, b, c, d = 39.72, 38.24, 42.75, 40.17
# # RoBERTaBase FreeLB TextFooler
# a, b, c, d = 30.75, 29.77, 36.81, 32.21

# # BERT-large FreeLB A2T
# a, b, c, d = 78.16, 78.16, 78.21, 78.04
# # BERT-large FreeLB PSO
# a, b, c, d = 65.49, 65.46, 65.56, 65.46
# # BERT-large FreeLB TextBugger
# a, b, c, d = 59.28, 59.35, 59.29, 59.27
# # BERT-large FreeLB TextFooler
# a, b, c, d = 55.34, 55.27, 55.26, 55.14

# # RoBERTaLarge  FreeLB A2T
# a, b, c, d = 70.38, 70.40, 71.30, 70.51
# # RoBERTaLarge  FreeLB PSO
# a, b, c, d = 56.23, 55.60, 57.20, 56.18
# # RoBERTaLarge  FreeLB TextBugger
# a, b, c, d = 44.52, 43.11, 44.42, 43.21
# # RoBERTaLarge  FreeLB TextFooler
# a, b, c, d = 37.56, 35.97, 38.59, 36.71





max_value = max(a, b, c, d)

# Corresponding labels
labels = ["P", "P-I", "P-D", "P-I-D"]

# Creating the bar chart
plt.figure(figsize=(5, 4))
bars = plt.bar(labels, [a, b, c, d], color=["blue", "green", "red", "purple"])


plt.axhline(y=max_value, color="black", linestyle="--")
# plt.text(x=-0.5, y=max_value - 2, s=f"Max Value: {max_value}", color="black")


# Adding titles and labels
# plt.title('Bar Chart Representation')
plt.xlabel("Control Type")
plt.ylabel("Accuracy")

# Display the bar chart
plt.show()
# %%
