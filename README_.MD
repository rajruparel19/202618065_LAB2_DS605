Vectorized Programming with NumPy & Data Wrangling with Pandas (Titanic Dataset)

Name:Raj Udaybhai Ruparel
Student ID:202618065
Dataset: Kaggle Titanic dataset (train.csv) — https://www.kaggle.com/c/titanic/data

About this project :

This is my submission for the NumPy + Pandas lab assignment. Everything is in one script, `titanic_analysis.py`, split into `# %%` cells so it can be run either as a normal Python file or opened in VS Code and stepped through cell by cell in the Interactive window. Part A covers NumPy basics and linear algebra, Part B is the actual Titanic data wrangling with Pandas.

I ran it fully on my own machine (not just relying on generated outputs) — figures get written to `figures/` and the cleaned dataset gets written to `data/train_cleaned.csv` every time the script runs.

What's covered

**Part A - NumPy**
- Task 1 - built a 100-value random int array with a fixed seed, pulled min/max/median/mean/std off it, made a second array with `arange`, played around with `zeros`/`ones`/`linspace`, and built out 2D and 3D arrays with indexing/slicing examples plus a reshape-then-flatten round trip.
- Task 2 - two 3x3 matrices, did addition, elementwise multiply, and real matrix multiplication with `@`. Also took the transpose and determinant of a square matrix, found its inverse, and checked `A @ A_inv` comes back to the identity matrix using `np.allclose`. No loops anywhere, all vectorized.
- Task 3 - generated 5000 points from a Normal(50, 10) distribution and compared the sample mean/std against the values I picked, then plotted a histogram.

**Part B - Titanic (Pandas)**
- Task 4 - loaded `train.csv`, ran `head`, `tail`, `shape`, `columns`, `info`, `describe`, and showed a `loc` example next to an `iloc` example so the difference is obvious.
- Task 5 - five filtering questions using boolean masks: older male passengers, female first-class survival rate, an age/fare/survival combo, solo travelers who didn't survive, and a Southampton fare comparison.
- Task 6 - grouped by Sex, by Pclass, by Sex+Pclass together, and by Embarked, pulling survival rates and averages out of each.
- Task 7 - checked missing values across every column and charted them, filled missing Age with the mean, then separately compared what mean/median/mode/random-value imputation would each do to the column, and worked out Fare outliers using the standard 1.5xIQR rule.
- Task 8 - added `FamilySize` and `IsAlone` columns, then built a pivot table of Sex x Pclass survival rates to see which group did best/worst.
- Task 9 - correlation heatmap, a survival-rate-by-sex bar chart, an Age vs Fare scatter split by survival, and my observations below.

## Files in this repo

- `titanic_analysis.py` - the main script, run this top to bottom
- `requirements.txt` - packages needed
- `data/train.csv` - the raw Titanic data
- `data/train_cleaned.csv` - what the script produces after cleaning (Age filled in, FamilySize/IsAlone added)
- `figures/` - the 5 PNGs the script generates
- `.gitignore` - keeps `venv/` out of the repo

## Running it

```
python -m venv venv
venv\Scripts\activate      (or source venv/bin/activate on Mac/Linux)
pip install numpy pandas matplotlib seaborn
python titanic_analysis.py
```

Heads up - a few matplotlib windows will pop up as it runs (one for the histogram, one for the missing-values chart, three more in Task 9). Just close each one and the script picks back up automatically.

## Observations

A few things stood out once I actually looked at the numbers:

- Being female mattered more than almost anything else. About 74% of women on board survived versus roughly 19% of men - that's a massive gap and lines up with the whole "women and children first" thing you hear about with the Titanic.
- Class mattered a lot too, and it stacks with sex. First class survival was around 63%, second class dropped to about 47%, and third class fell all the way to 24%. If you combine both factors, female first-class passengers survived almost every time (~97%), while third-class men barely made it out at all (~14%).
- Fare and Pclass are basically telling the same story - they're negatively correlated (makes sense, cheaper tickets = lower class), and Fare on its own has a decent positive relationship with survival too. Higher fare generally meant a better shot at surviving.
- Traveling completely alone hurt your odds a bit. People with no SibSp/Parch had a somewhat lower survival rate than people traveling with at least a small family.
- Age didn't correlate with survival nearly as strongly as I expected going in - the raw number is pretty weak. But looking at the Age vs Fare scatter plot, younger and higher-paying passengers do seem to cluster more toward the "survived" side.
- Fare has a bunch of outliers on the high end - running the IQR rule on it flags about 116 passengers (roughly 13% of everyone) as outliers, which tracks with a handful of wealthy families paying way more than everyone else for their tickets.
- Missing data is a real problem for Cabin specifically - it's missing on over 77% of rows, way more than Age (~20%) or Embarked (basically nothing missing there). Cabin would need serious feature engineering before it's usable for anything; Age is fine to just impute.

## Notes

- Used `np.random.seed(42)` everywhere randomness shows up so results are reproducible if you re-run it.
- Every NumPy operation is vectorized, no manual for-loops over array elements.
- Used the standard 891-row Titanic training set.
