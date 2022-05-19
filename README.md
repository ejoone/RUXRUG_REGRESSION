# An approach to interpretability in regression machine learning algorithm: a closer look at rule-based learning with Linear Programming

**BAQM Seminar - Team 7**

Please note that this code is written based on the code from sibirbil and the code for his paper "Discovering Classification Rules for Interpretable Learning with Linear Programming" (https://github.com/sibirbil/RuleDiscovery).

This code executes the RUX and RUG algorithms for the regression problem which are studied in the BAQM team 7 report "An approach to interpretability in regression machine learning algorithm: a closer look at rule-based learning with Linear Programming".
This paper extends the classification-focused rule-based interpretable machine learning algorithms based on Linear Programming (LP) to the regression problem. More specifically, we focus on Rule Extraction (RUX) and Rule Generation (RUG) algorithms by adjusting the mathematical formulations of these classification algorithms such that they can process regression data with a continuous target variable. Additionally, using nine datasets, this study assesses the accuracy and interpretability of RUX and RUG. Also, we perform a case study to compare the performance of RUX and RUG when they are applied to the datasets with different sample sizes and number of features. Overall, RUX and RUG algorithms are applicable to regression datasets and have an advantage over conventional tree-ensemble machine learning algorithms in terms of accuracy and interpretability. Our results have confirmed that RUX and RUG output significantly fewer rules compared to conventional tree-ensemble machine learning algorithms without sacrificing in terms of accuracy. Also, we conclude that RUX and RUG predictions are more accurate under datasets with larger sample sizes and datasets with fewer features. However, there is no objectively superior algorithm when comparing RUX with RUG. 


You can find the details in the submitted paper.

## Installation

 1. Install [Anaconda Distribution](https://www.anaconda.com/products/individual).

 2. Create a new environment and install the necessary packages:

 `conda create -n rulediscovery --channel=conda-forge python=3.8 numpy pandas scikit-learn cvxpy cvxopt gurobi`

 3. Activate the current environment:

 `conda acivate rulediscovery`

 4. Check whether the installation works:

 `python RUX_RUG_tests.py`

---

**OPTIONAL:**

To use the Gurobi solver, you need to first install
it. The solver is freely available for academic use. Check the
[related
page](https://www.gurobi.com/academia/academic-program-and-licenses/)
on Gurobi's website.
