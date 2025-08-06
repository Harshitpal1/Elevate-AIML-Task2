# Elevate AI & ML Internship: Task 2

## Project: Exploratory Data Analysis (EDA) on the Titanic Dataset

This repository contains the solution for Task 2 of the AI & ML Internship. The objective was to perform a comprehensive Exploratory Data Analysis (EDA) on the Titanic dataset to uncover insights about the factors that influenced survival.

### 1. Tools and Dataset

* **Tools Used:** Python, Pandas, Matplotlib, Seaborn
* **Dataset:** `titanic.csv` (The classic dataset containing passenger information from the Titanic).

### 2. Exploratory Data Analysis (EDA) Summary

The complete analysis is available in the `eda_titanic.ipynb` Jupyter Notebook. The key findings are summarized below:

#### **a. Summary Statistics**

I started by generating descriptive statistics using `df.describe()` to get a high-level overview of the numerical features.

* **Age:** The average age of passengers was around **29.7 years**. The age distribution is fairly wide, from infants (0.42 years) to elderly (80 years).
* **Fare:** The fare paid by passengers varied dramatically, with a mean of **$32.20** but a maximum of **$512.33**. This indicates a significant skew, with a few passengers paying very high fares.
* **Survival:** Approximately **38.4%** of the passengers in this dataset survived.

#### **b. Univariate Analysis (Analyzing Single Features)**

* **Survival Rate:** A count plot showed that more passengers died than survived.
* **Passenger Class (Pclass):** The majority of passengers were in the **3rd class**, which was the most populated.
* **Gender (Sex):** There were significantly more **male** passengers than female passengers.
* **Age Distribution:** A histogram of the 'Age' column revealed a right-skewed distribution, with a large number of passengers in the 20-30 age group.

#### **c. Bivariate Analysis (Analyzing Relationships)**

I created several visualizations to explore the relationships between different features and the survival outcome.

* **Survival by Gender:** A bar chart clearly showed that **females had a much higher survival rate (around 74%)** compared to males (around 19%). This is one of the strongest indicators of survival.
* **Survival by Passenger Class:** A bar chart demonstrated that **1st class passengers had the highest survival rate (over 60%)**, while 3rd class passengers had the lowest (below 25%). This suggests a strong link between socio-economic status and survival.
* **Survival by Age:** A box plot and a histogram layered by survival status showed that **children (age < 10) had a higher survival rate**. Very elderly passengers had a lower chance of survival.

#### **d. Correlation Analysis**

* **Correlation Matrix:** I generated a heatmap of the correlation matrix for all numerical features.
* **Key Finding:** There was a notable **negative correlation between 'Pclass' and 'Fare'**, meaning passengers in higher classes (lower Pclass number) paid higher fares, which is logical. There was no single overwhelmingly strong correlation with 'Survived', indicating that survival was likely influenced by a combination of factors.

---

## 3. Interview Questions and Answers

**1. What is the purpose of EDA?**
The primary purpose of EDA is to **understand the data** before modeling. It involves summarizing the main characteristics of a dataset, often with visual methods, to uncover patterns, spot anomalies, test hypotheses, and check assumptions. It helps in making informed decisions about data cleaning, feature engineering, and choosing the right machine learning model.

**2. How do boxplots help in understanding a dataset?**
Boxplots are excellent for understanding the **distribution of numerical data**. They show key statistical measures in one visual:
* The **median** (the central line).
* The **interquartile range (IQR)** (the box), which represents the middle 50% of the data.
* The **whiskers**, which show the range of the data.
* **Outliers**, which are plotted as individual points.
This makes them very effective for comparing distributions across different categories (e.g., comparing the age of survivors vs. non-survivors).

**3. What is correlation and why is it useful?**
Correlation is a statistical measure that expresses the extent to which two variables are linearly related, meaning they change together at a constant rate. It's useful because it helps us understand the **relationships between features**. For example, a high correlation between two features might indicate redundancy (multicollinearity), which can be problematic for some machine learning models.

**4. How do you detect skewness in data?**
Skewness refers to the asymmetry in a statistical distribution. I detect it in two main ways:
* **Visually:** By plotting a **histogram** or a density plot. A long tail on the right indicates positive (right) skew, and a long tail on theleft indicates negative (left) skew.
* **Statistically:** By using the `.skew()` method in Pandas. A skewness value greater than 1 or less than -1 is generally considered highly skewed.

**5. What is multicollinearity?**
Multicollinearity is a phenomenon in which one predictor variable in a multiple regression model can be linearly predicted from the others with a substantial degree of accuracy. In simpler terms, it's when two or more independent variables are highly correlated. This can be a problem because it undermines the statistical significance of an independent variable.

**6. What tools do you use for EDA?**
I primarily use Python with the following libraries:
* **Pandas:** For data manipulation and generating descriptive statistics.
* **Matplotlib & Seaborn:** For creating a wide range of static visualizations like histograms, box plots, bar charts, and heatmaps.
* **Plotly:** For creating interactive visualizations, which can be very useful for deeper exploration.

**7. Can you explain a time when EDA helped you find a problem?**
During this Titanic EDA, I discovered that the 'Age' column had 177 missing values. If I had proceeded directly to modeling, this would have caused an error or led to a biased model. The EDA process forced me to notice this and develop a strategy to handle the missing data, such as imputing the missing ages based on the mean or median.

**8. What is the role of visualization in ML?**
Visualization plays a critical role throughout the machine learning lifecycle:
* **During EDA:** To understand data distributions, relationships, and patterns.
* **During Model Evaluation:** To visualize model performance, such as plotting a confusion matrix, ROC curve, or learning curves.
* **During Results Presentation:** To communicate complex findings to stakeholde
