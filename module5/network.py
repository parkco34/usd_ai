#!/usr/bin/env python
"""
REFERENCES:
    https://medium.com/@ugursavci/complete-exploratory-data-analysis-using-python-9f685d67d1e4
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import dedent
import networkx as nx
from pgmpy.estimators import HillClimbSearch, BicScore, PC, BayesianEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# sklearn (for realistic performance)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
df = pd.read_csv("heart_failure_prediction.csv")

# 1) EDA =========================
# Cholesterol shouldn't be zero: Replace zeros with median value
df["Cholesterol"] = df["Cholesterol"].replace(0,df[df["Cholesterol"] > 0]["Cholesterol"].median())

# Resting blood pressure cannot be zero
df["RestingBP"] = df["RestingBP"].replace(0, df[df["RestingBP"] > 0]["RestingBP"].median())

# Segmant and sort data values into bins
# Bins for Age (since continuous, numeric data)
df["Age_bin"] = pd.cut(
    df["Age"], 
    bins=[0, 39, 59, 100], 
    labels=["<40", "40-59", "60+"])

# Oldpeak bins
df["Oldpeak_bin"] = pd.cut(
    df["Oldpeak"], 
    bins=[-np.inf, 1, 2, np.inf], 
    labels=["Normal", "Moderate", "High"])

# Cholesterol bins
df["Cholesterol_bin"] = pd.cut(
    df["Cholesterol"],
    bins=[0, 200, 240, np.inf],
    labels=["Normal", "Borderline", "High"]
)

# Max heart rate bins
q1, q2 = df["MaxHR"].quantile([0.33, 0.66])
df["MaxHR_bin"] = pd.cut(
    df["MaxHR"],
    bins=[-np.inf, q1, q2, np.inf],
    labels=["Low", "Medium", "High"]
)

# Dataset Overview and Descriptive Statistics (What am I working with)
print("======== Preview of Data ======== ")
print(f"\nSample of the data:\n{df.head()}")
print(f"\n(Rows, Columns) = {df.shape}")
df.info()
print(f"\nDescribe (numeric data): {df.describe()}")
print("The top is the most common value. ")
print("The freq is the most common value’s frequency.")
print(f"\nNumber of duplicated rows: {df.duplicated().sum()}")

# Plotting
plt.figure(figsize=(10 , 6))
ax = plt.gca()

# ======================+++++++====================
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
axes = axes.ravel()  # flatten to 1D for easy indexing

# 1) Heart disease by age group
ax = axes[0]
pd.crosstab(df["Age_bin"], df["HeartDisease"], normalize="index")[1].plot(
    kind="bar", ax=ax
)
ax.set_title("P(Heart Disease | Age Group)")
ax.set_ylabel("Probability")
ax.set_xlabel("Age_bin")

# 2) ST Slope vs Heart Disease
ax = axes[1]
pd.crosstab(df["ST_Slope"], df["HeartDisease"], normalize="index")[1].plot(
    kind="bar", ax=ax
)
ax.set_title("P(Heart Disease | ST Slope)")
ax.set_ylabel("Probability")
ax.set_xlabel("ST_Slope")

# 3) Exercise Angina vs Heart Disease
ax = axes[2]
pd.crosstab(df["ExerciseAngina"], df["HeartDisease"], normalize="index")[1].plot(
    kind="bar", ax=ax
)
ax.set_title("P(Heart Disease | Exercise Angina)")
ax.set_ylabel("Probability")
ax.set_xlabel("ExerciseAngina")

# 4) Cholesterol x MaxHR risk heatmap: P(HeartDisease=1 | Chol_bin, MaxHR_bin)
ax = axes[3]

chol_order = ["Normal", "Borderline", "High"]
hr_order = ["Low", "Medium", "High"]

heat = pd.crosstab(
    df["Cholesterol_bin"],
    df["MaxHR_bin"],
    values=df["HeartDisease"],
    aggfunc="mean",
).reindex(index=chol_order, columns=hr_order)

# If any bin-combo has no samples -> NaN; fill for plotting
heat = heat.fillna(0)

im = ax.imshow(
    heat.values,
    origin="lower",
    aspect="auto",
    vmin=0,
    vmax=1,
    cmap="Reds",
)

ax.set_xticks(range(len(heat.columns)))
ax.set_xticklabels(heat.columns)
ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels(heat.index)
ax.set_title("Risk Heatmap: Cholesterol_bin × MaxHR_bin")
ax.set_xlabel("MaxHR_bin")
ax.set_ylabel("Cholesterol_bin")

# annotate each cell
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        ax.text(j, i, f"{heat.iat[i, j]:.2f}", ha="center", va="center")

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(Heart Disease)")

plt.tight_layout()
plt.show()

# Interpretation
print(dedent(f"""
EDA:
The ST_Slope and ExerciseAngina are the most profound indicators of heart disease risk.
             ST_Slope='Flat' pushes the probability to ~{heat.max().max():.2f} while 'Up' drops it to roughly 0.15, and ExerciseAngina='Y' almost triples the risk compared to 'N' (NO).
Age shows an obvious trend with the risk rising from ~0.32 to ~0.65 for younger and older (60+) people, respectively.
The Cholesterol and Max Heart Rate heatmap shows that patients with high cholesterol and low heart rate are associated with higher risk of heart disease.  
You'll notice that the color intensity isn't monotonic for cholesterol alone.  This is because the heat map is representing a 'Conditional' pattern.

Overall, excersie-related features like MaxHR, ExerciseAngina, etc.
are more useful for assessing heart disease risk than cholesterol alone.
             """))

def learn_structure_pc(dataframe):
    """
    Learns BN structure using PC algorithm (constant-based) via
    Removing edges when conditional independence is detected w/ chi-square and G-test style tests on variables.
    Then orient edges to form consistent DAG.
    ------------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame) Dataframe to work with

    OUTPUT:
        model: (BayesianNetwork) DAG structure
    """
    pc = PC(dataframe)
    # Stable for reproducibility
    skel, sep_sets = pc.build_skeleton(variant="stable")
    pdag = pc.skeleton_to_pdag(skel, sep_sets)
    dag = pc.pdag_to_dag(pdag)
    
    return BayesianNetwork(dag.edges())

def plot_structure(model, title):
    """
    Visualize learned Bayesian Network Structure
    ----------------------------------------------
    INPUT:
        model: (Bayesian Network)
        title: (str) Title of plot
    """
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=7)
    nx.draw(G, pos, with_labels=True, node_size=1600, font_size=9, arrows=True)
    plt.title(title)
    plt.show()

# 4) ================= Parameter Learning  =================
def fit_cpts(model, dataframe):
    """
    Fits CPTs (Conditional probability tables) using Bayesian parameter estimation (Dirichlet smoothing) to avoid zero-probability CPT entries harmful to inference.  
    ----------------------------------------------------------
    INPUT:
        model: (Bayesian Network)
        dataframe: (pd.DataFrame) 

    OUTPUT:
        fitted: (Bayesian network structure with CPTs)
    """
    # Create copy of model
    fitted = model.copy()
    fitted.fit(dataframe, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)

    return fitted

def disease_prob(fitted, evidence):
    """
    Computes P(HeartDisease=1 | evidence) via inference.
    ------------------------------------------------
    INPUT:
        fitted: (?)
        evidence: (?)

    OUTPUT:
        posterior: (float) Posterior probability of having heart diease
    """
    # Converts CPTs into factors and eliminates hidden variables
    infer = VariableElimination(fitted)
    # Get distribution over states of HeartDisease
    q = infer.query(variables=["HeartDisease"], evidence=evidence, show_progress=False)

    # States are strings '0' and '1'
    posterior = float(q.values[list(q.state_names["HeartDisease"]).index("1")])

    return posterior

def run_query(fitted):
    """
    Runs five clinical querys for assignment w/ interpretaion.
    """
    # Baseline
    p_base = disease_prob(fitted, evidence={})
    print(f"\nBaseline P(HeartDisease=1) =  {p_base:.3f}\n")

    # a)
    e_a = {"Age_bin": "60+", "ST_Slope": "Flat"}
    p_a = disease_prob(fitted, e_a)

    print(f"(a) P(HD=1 | Age_bin=60+, ST_Slope=Flat) = {p_a:.3f}")
    print(dedent("""
Compared ot baseline this shows the model's estimated risk for an older patient with a flat ST segment.
If this is well above baseline, the BN attributes increased risk to these observed findings.\n
                 """))

    # b) w/ ExerciseAngina=1
    e_b = {"Age_bin": "60+", "ST_Slope": "Flat", "ExerciseAngina": "Y"}
    p_b = disease_prob(fitted, e_b)
    
    print(dedent(f"""b)
P(HeartDisease=1 | Age_bin=60+, ST_Slope=Flat, ExerciseAngina=1) = {p_b:.3f}
                 """))
    print(dedent("""
\nThe risk should go up when adding exercise-angina if the Bayesian Network learned it as strong predictor.
                """))

    # c)
    e_c = {"Cholesterol_bin": "High", "MaxHR_bin": "Low"}
    p_c = disease_prob(fitted, e_c)
    print(dedent(f"""(c) 
                 P(HeartDisease=1 | Cholesterol_bin=High, MaxHR_bin=Low) = {p_c:.3f}
                 """))

    print(dedent("""
Including this metabolic risk (high cholesterol and low max HR), if the posterior value is increased, the Bayesian Network treatscthese two as a high-risk factor.
                 """))

    # d)
    e_d = {"ChestPainType": "ATA", "ExerciseAngina": "N"}
    p_d = disease_prob(fitted, e_d)
    print(f"(d) P(HD=1 | ChestPainType=ATA, ExerciseAngina=0) = {p_d:.3f}")
    print(dedent("""
In this case, we're looking at a scenario with less alarming chest pain with no exercised-induced angina.

If this is closer to or below baseline the BN is consistent with a
lower-risk.\n
                 """))

    # e) Full diagnostic
    e_e = {"Age_bin": "60+", "ST_Slope": "Flat", "ExerciseAngina": "1", "Oldpeak_bin": "High"}
    p_e = disease_prob(fitted, e_e)
    print(f"(e) P(HD=1 | Age_bin=60+, ST_Slope=Flat, ExerciseAngina=1, Oldpeak_bin=High) = {p_e:.3f}")

    print(dedent("""
This stacks multiple high-risk indicators, where the posterior should focus on higher risk if learned associations match expectations, clinically.

This query is explainable: you can report how each added evidence shifts the
                 probability.\n
                 """))

# 5) ============ BN for Classifcation ============ 
def predict_prob_rows(fitted, dataframe, features):
    infer = VariableElimination(fitted)
    probs = []

    for i, row in dataframe.iterrows():
        evidence = {c: str(row[c]) for c in features}
        q = infer.query(variables=["HeartDisease"], evidence=evidence, show_progress=False)
        p1 = float(q.values[list(q.state_names["HeartDisease"]).index("1")])
        probs.append(p1)

    return np.array(probs)

def eval_classifier(dataframe, final_struct):
    """
    Train/test split evaluation.
    Fits CPTs on train, predicts proababilities on test, then predicts class.
Reports accuracy and ROC AUC
    ------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame)
        final_struct: (BayesianNetwork) The Final structure

    OUTPUT:
        None
    """
    # Observed Features
    feats = [thing for thing in dataframe.columns if thing != "HeartDisease"]

    # Train/test split
    train_df, test_df = train_test_split(
        dataframe, 
        test_size=0.30,
        random_state=7,
        stratify=dataframe["HeartDisease"])
    
    fitted = fit_cpts(final_struct, train_df)

    # Predict on test
    X_test = test_df[feats]
    y_test = test_df["HeartDisease"].astype(int)

    probs = predict_prob_rows(fitted, X_test, feats)
    y_hat = (probs > 0.5).astype(int)

    # AUC and accuracy
    acc = accuracy_score(y_test, y_hat)
    auc = roc_auc_score(y_test, probs)

    # Output
    print(f"Accuracy: {acc:.3f}")
    print(f"AUC: {auc:.3f}\n")

# Execution
print(dedent("""
Risk-factor summary (EDA):
- Older age groups (especially 60+) tend to show higher empirical P(HeartDisease=1).
- ST_Slope='Flat' and 'Down' typically correspond to higher risk than 'Up' in this dataset.
- Exercise-induced angina is usually strongly associated with HeartDisease (large conditional rate shift).
- High Cholesterol combined with Low MaxHR forms a high-risk corner in the heatmap (interaction-style signal).
"""))

# 3) ========= Structure Learning =========
# Learns structure via HIll-climbing search 
hc = HillClimbSearch(df)
best = hc.estimate(scoring_method=BicScore(df))
bn1 = BayesianNetwork(best.edges())

# Structured learning via PC algorithm
bn2 = learn_structure_pc(df)

# Plotting
plot_structure(bn1, "Structure A: HIll-Climb + BIC")
plot_structure(bn2, "Structure B: PC Algorithm")

# Choose  final structure
def edge_count(model):
    return len(model.edges())

print("Structure Comparison")
print(f"Edges in bn1 (HC + BIC): {edge_count(bn1)}")
print(f"Edges in bn2 (PC): {edge_count(bn2)}")

# Preferred structure: Hill-climbing
final_struct = bn1

print(dedent(f"""
Final Structure Choice:
I'll use Structure A Hill-climb + BIC for the rest of the project as it typcially produces parsimonious newtwork by penalizing over more complex graphs (BIC).

Resulting dependencies easier to interpret.
             """))

# Fit CPTs on full data and run
fitted_full = fit_cpts(final_struct, df)
run_query(fitted_full)

# Classifier evaluation
eval_classifier(df, final_struct)

print(dedent("""\n
WHY A BN OVER A SLIGHTLY MORE ACCURATE BLACK-BOX?
A clinic could prefer a Bayesian Network because it's explicitly explainable (interpretable CPTs).

Even with missing input data, it still supports meaningful clinical
probabilistic inferences.
This improves trust compared to black-box models that might be difficult to justify for an individual.

REAL-WORLD EXAMPLE:
Bayesian Health Inc. deployed clinical risk prediction systems used for tasks like sepsis detection in healthcare settings.

REFERENCES
https://newsroom.clevelandclinic.org/2025/09/23/cleveland-clinic-announces-the-expanded-rollout-of-bayesian-healths-ai-platform-for-sepsis-detection
             """))








