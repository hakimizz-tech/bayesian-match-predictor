# Turkish Süper Lig Predictive Modeling

> Bridging Statistical Theory and ML Systems Design

---

This project tackles the complex challenge of forecasting football match outcomes in the **Turkish Süper Lig**. Rather than applying algorithms blindly to raw data, this notebook treats predictive modeling as a rigorous engineering process intertwined with fundamental statistical theory. By meticulously engineering temporal features, auditing for data leakage, and validating probabilistic assumptions, the analysis establishes a robust pipeline suitable for real-world, production-level predictions.

## Core Objectives and Achievements

The primary objective of this project was to design a machine learning workflow that respects the **temporal nature** of sports data while uncovering the statistical realities of goal-scoring.

The project successfully achieved:

- A **leakage-free feature engineering pipeline** that accurately tracks historical team performance up to the exact moment before a match begins.
- Demonstrated that **engineered, mathematically grounded features** (like rolling win rates and point accumulations) significantly outperform raw, unadjusted data in driving the model's predictive power.

---

## Methodological Decisions and Implementation

### Chronological Sequencing and Data Leakage Prevention

In any time-series or sequential predictive task, the greatest threat to model validity is **look-ahead bias** — the inadvertent inclusion of future information in the training set. A strict chronological approach was mandated to simulate a real-world production environment where a model must predict tomorrow's match knowing only today's data.

The dataset was strictly sorted by date. More importantly, when calculating historical performance metrics — such as a team's historical win rate or average goals scored — a **shifting mechanism** was explicitly applied. The historical win rate at match $t$ is calculated only using matches from $1$ to $t-1$:

$$W_{t} = \frac{1}{t-1} \sum_{i=1}^{t-1} I(O_i = \text{Win})$$

where $I$ is an indicator function that equals $1$ if the historical match resulted in a win, and $0$ otherwise.

To guarantee the integrity of this operation, a dedicated **leakage differential audit** was performed. By comparing the calculated historical rates against the actual match outcomes and measuring the discrepancy, the pipeline mathematically proved the absence of target leakage before any model training commenced.

### Statistical Validation of Target Distributions

Before attempting to predict the exact number of goals scored by a home team (*Full Time Home Goals*), it was necessary to understand the underlying statistical distribution of the data. Count data, such as goals in a football match, rarely conform to a Gaussian (normal) distribution.

The data was empirically tested against a theoretical **Poisson distribution**. The assumption is that goals occur independently at a constant average rate over the fixed time interval of a 90-minute match. The probability of observing $k$ goals is given by:

$$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

where $\lambda$ represents the historical average of home goals scored across the dataset.

By visually and statistically plotting the actual density of home goals against this theoretical probability mass function, the project validated that a Poisson-like distribution accurately models the target. This decision ensures that any subsequent regression modeling is grounded in correct probabilistic assumptions, preventing the model from predicting impossible negative values or misinterpreting the variance.

### Class Imbalance and the Home Advantage

Football is inherently imbalanced due to the well-documented **"home advantage."** Evaluating the categorical outcomes — *Home Win*, *Draw*, *Away Win* — revealed a skewed distribution heavily favoring the home side.

Recognizing this imbalance was a critical decision point. It dictates that relying on a naive metric like simple classification accuracy would be highly misleading; a model could achieve artificially high accuracy simply by always predicting a home win. This structural reality informed the approach to model evaluation, highlighting the need for **nuanced metrics** that account for minority classes, particularly the notoriously difficult-to-predict *Draw*.

### Feature Importance and Model Drivers

The culmination of the predictive pipeline involved analyzing what the machine learning model actually learned. The evaluation of relative feature importance highlighted that the model heavily prioritized the **heavily engineered, mathematical features** over raw data points.

The metrics derived from the shifting windows — such as accumulated points and precise rolling win rates — dominated the decision trees. This confirms the underlying hypothesis of the project:

> *In highly stochastic environments like sports, a model's predictive power is bounded entirely by the quality, temporal accuracy, and statistical validity of the features engineered for it.*