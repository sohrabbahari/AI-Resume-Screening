import pandas as pd

#Load the Resume dataset
df = pd.read_csv(r"C:\Users\sohra\Desktop\Projects\ResumeScreeningAI\dataset\Resume.csv")


# Display the first 5 rows
print("First 5 Row of Dataset:")
print(df.head())

# Check dataset info
print("\n Dataset Information:")
print(df.info())

# show all unique job categories
print("\n Unique Job Categories in Dataset:")
print(df["Category"].unique())

# Count number of resumes in each category
print("\n Number of Resume per Job Category:")
print(df["Category"].value_counts())
