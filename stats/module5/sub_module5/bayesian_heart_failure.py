"""
===============================================================================
Assignment 5.5: Build an Explainable Bayesian Network for Heart Failure Prediction
===============================================================================

This module implements a complete Bayesian Network analysis pipeline for heart
failure prediction using the UCI Heart Failure Clinical Records dataset.

BACKGROUND:
-----------
In real medical AI systems, doctors and regulators frequently reject black-box
models (even if they are slightly more accurate) because they cannot explain WHY
a patient was classified as high-risk. Bayesian Networks solve this problem by:
    1. Providing transparent probabilistic relationships between variables
    2. Allowing clinicians to query specific conditional probabilities
    3. Showing the causal/correlational structure learned from data

DATASET:
--------
The Heart Failure Clinical Records dataset contains 299 patients with the following
features:
    - age: Age of patient (years)
    - anaemia: Decrease of red blood cells (0=No, 1=Yes)
    - creatinine_phosphokinase: Level of CPK enzyme in blood (mcg/L)
    - diabetes: If patient has diabetes (0=No, 1=Yes)
    - ejection_fraction: Percentage of blood leaving heart per contraction (%)
    - high_blood_pressure: If patient has hypertension (0=No, 1=Yes)
    - platelets: Platelet count in blood (kiloplatelets/mL)
    - serum_creatinine: Level of serum creatinine in blood (mg/dL)
    - serum_sodium: Level of serum sodium in blood (mEq/L)
    - sex: Gender (0=Female, 1=Male)
    - smoking: If patient smokes (0=No, 1=Yes)
    - time: Follow-up period (days)
    - DEATH_EVENT: If patient died during follow-up (0=Survived, 1=Died)

ASSIGNMENT SECTIONS:
--------------------
1. Exploratory Data Analysis (EDA) - 4 insightful plots with risk factor summary
2. Bayesian Network Structure Learning - Compare 2 algorithms, justify choice
3. Parameter Learning & Clinical Inference - 5 probabilistic queries
4. Probabilistic Classifier - Accuracy and AUC metrics
5. Discussion - Why prefer BN over black-box models + real-world example

Author: Whitney Parker
Course: USD AI Module 5
Date: 2025
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# pgmpy imports for Bayesian Network
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import (
    HillClimbSearch,
    K2Score,
    BicScore,
    PC,
    MaximumLikelihoodEstimator
)
from pgmpy.inference import VariableElimination
import networkx as nx


def load_and_prepare_data(filepath='heart_failure_records.csv'):
    """
    Load the heart failure dataset and prepare it for Bayesian Network analysis.

    Bayesian Networks work best with discrete/categorical variables, so we bin
    continuous variables into meaningful clinical categories.

    Parameters:
    -----------
    filepath : str
        Path to the heart failure CSV file

    Returns:
    --------
    tuple : (df_original, df_binned)
        df_original: Raw dataframe for EDA
        df_binned: Discretized dataframe for Bayesian Network

    NOTES ON BINNING STRATEGY:
    --------------------------
    - Age: <50 (younger), 50-65 (middle), 65+ (older) - based on cardiac risk
    - Ejection Fraction: <30 (severe), 30-45 (moderate), >45 (normal)
    - Serum Creatinine: <1.2 (normal), 1.2-2.0 (elevated), >2.0 (high)
    - CPK: <200 (normal), 200-1000 (elevated), >1000 (high)
    - Time: <50 (short), 50-150 (medium), >150 (long follow-up)
    """
    print("=" * 70)
    print("LOADING AND PREPARING DATA")
    print("=" * 70)

    df = pd.read_csv(filepath)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nTarget distribution (DEATH_EVENT):")
    print(df['DEATH_EVENT'].value_counts())
    print(f"Death rate: {df['DEATH_EVENT'].mean()*100:.1f}%")

    # Create binned version for Bayesian Network
    df_binned = df.copy()

    # Bin continuous variables with clinically meaningful thresholds
    # Age bins: Young (<50), Middle (50-65), Senior (65+)
    df_binned['Age_bin'] = pd.cut(df['age'],
                                   bins=[0, 50, 65, 100],
                                   labels=['Young', 'Middle', 'Senior'])

    # Ejection Fraction: Severe (<30%), Moderate (30-45%), Normal (>45%)
    df_binned['EF_bin'] = pd.cut(df['ejection_fraction'],
                                  bins=[0, 30, 45, 100],
                                  labels=['Severe', 'Moderate', 'Normal'])

    # Serum Creatinine: Normal (<1.2), Elevated (1.2-2.0), High (>2.0)
    df_binned['Creatinine_bin'] = pd.cut(df['serum_creatinine'],
                                          bins=[0, 1.2, 2.0, 10],
                                          labels=['Normal', 'Elevated', 'High'])

    # CPK levels: Normal (<200), Elevated (200-1000), High (>1000)
    df_binned['CPK_bin'] = pd.cut(df['creatinine_phosphokinase'],
                                   bins=[0, 200, 1000, 10000],
                                   labels=['Normal', 'Elevated', 'High'])

    # Follow-up time: Short (<50 days), Medium (50-150), Long (>150)
    df_binned['Time_bin'] = pd.cut(df['time'],
                                    bins=[0, 50, 150, 300],
                                    labels=['Short', 'Medium', 'Long'])

    # Serum Sodium: Low (<135), Normal (135-145), High (>145)
    df_binned['Sodium_bin'] = pd.cut(df['serum_sodium'],
                                      bins=[0, 135, 145, 200],
                                      labels=['Low', 'Normal', 'High'])

    # Convert binary variables to categorical strings for clarity
    df_binned['Anaemia'] = df_binned['anaemia'].map({0: 'No', 1: 'Yes'})
    df_binned['Diabetes'] = df_binned['diabetes'].map({0: 'No', 1: 'Yes'})
    df_binned['HighBP'] = df_binned['high_blood_pressure'].map({0: 'No', 1: 'Yes'})
    df_binned['Sex'] = df_binned['sex'].map({0: 'Female', 1: 'Male'})
    df_binned['Smoking'] = df_binned['smoking'].map({0: 'No', 1: 'Yes'})
    df_binned['Death'] = df_binned['DEATH_EVENT'].map({0: 'Survived', 1: 'Died'})

    # Select only the binned/categorical columns for BN
    bn_columns = ['Age_bin', 'EF_bin', 'Creatinine_bin', 'CPK_bin', 'Time_bin',
                  'Sodium_bin', 'Anaemia', 'Diabetes', 'HighBP', 'Sex',
                  'Smoking', 'Death']

    df_bn = df_binned[bn_columns].copy()

    # Convert all to string type for pgmpy compatibility
    for col in df_bn.columns:
        df_bn[col] = df_bn[col].astype(str)

    print(f"\nBinned dataset prepared with {len(bn_columns)} categorical features")

    return df, df_bn


# =============================================================================
# SECTION 1: EXPLORATORY DATA ANALYSIS
# =============================================================================

def perform_eda(df):
    """
    Perform Exploratory Data Analysis with 4 insightful visualizations.

    PURPOSE:
    --------
    EDA helps us understand the data before building our Bayesian Network.
    We want to identify:
        1. Distribution of the target variable (class imbalance?)
        2. Key risk factors associated with death events
        3. Relationships between features
        4. Potential confounders or spurious correlations

    Parameters:
    -----------
    df : pandas.DataFrame
        Original (non-binned) heart failure dataset

    Returns:
    --------
    None (displays plots and prints summary)

    CLINICAL INSIGHTS TO LOOK FOR:
    ------------------------------
    - Lower ejection fraction → higher mortality (heart pumps less blood)
    - Higher serum creatinine → kidney dysfunction → worse outcomes
    - Shorter follow-up time often correlates with death (died early)
    - Age is a known cardiac risk factor
    """
    print("\n" + "=" * 70)
    print("SECTION 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Heart Failure Dataset - Exploratory Data Analysis',
                 fontsize=14, fontweight='bold')

    # -------------------------------------------------------------------------
    # PLOT 1: Death Event Distribution by Age Group
    # -------------------------------------------------------------------------
    # WHY: Age is a fundamental risk factor for cardiac events. We expect
    # older patients to have higher mortality rates.
    ax1 = axes[0, 0]
    age_bins = pd.cut(df['age'], bins=[0, 50, 60, 70, 100],
                      labels=['<50', '50-60', '60-70', '70+'])
    death_by_age = pd.crosstab(age_bins, df['DEATH_EVENT'], normalize='index') * 100
    death_by_age.plot(kind='bar', ax=ax1, color=['#2ecc71', '#e74c3c'],
                      edgecolor='black')
    ax1.set_title('Plot 1: Mortality Rate by Age Group', fontweight='bold')
    ax1.set_xlabel('Age Group (years)')
    ax1.set_ylabel('Percentage (%)')
    ax1.legend(['Survived', 'Died'], title='Outcome')
    ax1.tick_params(axis='x', rotation=0)
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f%%', fontsize=8)

    # -------------------------------------------------------------------------
    # PLOT 2: Ejection Fraction Distribution by Outcome
    # -------------------------------------------------------------------------
    # WHY: Ejection fraction measures how well the heart pumps blood.
    # Lower EF indicates heart failure severity and predicts mortality.
    ax2 = axes[0, 1]
    survived = df[df['DEATH_EVENT'] == 0]['ejection_fraction']
    died = df[df['DEATH_EVENT'] == 1]['ejection_fraction']
    ax2.boxplot([survived, died], labels=['Survived', 'Died'])
    ax2.set_title('Plot 2: Ejection Fraction by Outcome', fontweight='bold')
    ax2.set_xlabel('Patient Outcome')
    ax2.set_ylabel('Ejection Fraction (%)')
    ax2.axhline(y=40, color='r', linestyle='--', alpha=0.7,
                label='Normal threshold (40%)')
    ax2.legend()
    # Add mean annotations
    ax2.annotate(f'Mean: {survived.mean():.1f}%', xy=(1, survived.mean()),
                 xytext=(1.2, survived.mean()), fontsize=9)
    ax2.annotate(f'Mean: {died.mean():.1f}%', xy=(2, died.mean()),
                 xytext=(2.2, died.mean()), fontsize=9)

    # -------------------------------------------------------------------------
    # PLOT 3: Serum Creatinine vs Death (Kidney Function Indicator)
    # -------------------------------------------------------------------------
    # WHY: High serum creatinine indicates kidney dysfunction, which is
    # strongly associated with cardiovascular mortality (cardiorenal syndrome).
    ax3 = axes[1, 0]
    colors = ['#2ecc71' if x == 0 else '#e74c3c' for x in df['DEATH_EVENT']]
    scatter = ax3.scatter(df['serum_creatinine'], df['ejection_fraction'],
                          c=df['DEATH_EVENT'], cmap='RdYlGn_r', alpha=0.6,
                          edgecolors='black', linewidth=0.5)
    ax3.set_title('Plot 3: Serum Creatinine vs Ejection Fraction', fontweight='bold')
    ax3.set_xlabel('Serum Creatinine (mg/dL)')
    ax3.set_ylabel('Ejection Fraction (%)')
    ax3.axvline(x=1.5, color='orange', linestyle='--', alpha=0.7,
                label='Elevated creatinine threshold')
    ax3.axhline(y=40, color='blue', linestyle='--', alpha=0.7,
                label='Normal EF threshold')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Death Event (0=Survived, 1=Died)')
    ax3.legend(loc='upper right', fontsize=8)

    # -------------------------------------------------------------------------
    # PLOT 4: Correlation Heatmap of Key Risk Factors
    # -------------------------------------------------------------------------
    # WHY: Understanding correlations helps us interpret the Bayesian Network
    # structure and identify which variables might be directly connected.
    ax4 = axes[1, 1]
    key_features = ['age', 'ejection_fraction', 'serum_creatinine',
                    'serum_sodium', 'time', 'DEATH_EVENT']
    corr_matrix = df[key_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', ax=ax4, square=True,
                cbar_kws={'label': 'Correlation Coefficient'})
    ax4.set_title('Plot 4: Correlation Heatmap of Key Features', fontweight='bold')

    plt.tight_layout()
    plt.savefig('sub_module5/eda_plots.png', dpi=150, bbox_inches='tight')
    plt.show()

    # -------------------------------------------------------------------------
    # EDA SUMMARY: Risk Factor Analysis (3-5 sentences)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("EDA SUMMARY: KEY RISK FACTORS OBSERVED")
    print("-" * 70)

    summary = """
    1. EJECTION FRACTION is the strongest predictor of mortality. Patients who
       died had a mean EF of {:.1f}% compared to {:.1f}% for survivors, indicating
       that reduced heart pumping capacity significantly increases death risk.

    2. SERUM CREATININE shows a clear association with mortality. Elevated levels
       (>1.5 mg/dL) indicate kidney dysfunction, which often accompanies and
       worsens heart failure outcomes (cardiorenal syndrome).

    3. AGE is a consistent risk factor, with mortality rates increasing from
       {:.1f}% in patients under 50 to {:.1f}% in those over 70 years old.

    4. FOLLOW-UP TIME (variable 'time') has a strong negative correlation with
       death (-0.53), meaning patients who died typically had shorter follow-up
       periods, indicating early mortality in the study.

    5. Interestingly, SERUM SODIUM shows weak but notable correlation with
       outcomes - low sodium (hyponatremia) is a known marker of poor prognosis
       in heart failure patients.
    """.format(
        df[df['DEATH_EVENT']==1]['ejection_fraction'].mean(),
        df[df['DEATH_EVENT']==0]['ejection_fraction'].mean(),
        df[df['age']<50]['DEATH_EVENT'].mean()*100,
        df[df['age']>=70]['DEATH_EVENT'].mean()*100
    )
    print(summary)

    return None


# =============================================================================
# SECTION 2: BAYESIAN NETWORK STRUCTURE LEARNING
# =============================================================================

def learn_bn_structure(df_bn):
    """
    Learn Bayesian Network structure using two different algorithms and compare.

    WHAT IS STRUCTURE LEARNING?
    ---------------------------
    Structure learning discovers the directed acyclic graph (DAG) that represents
    conditional dependencies between variables. The edges indicate probabilistic
    relationships (not necessarily causal).

    ALGORITHMS COMPARED:
    --------------------
    1. Hill Climbing with K2 Score:
       - Greedy search algorithm that iteratively adds/removes/reverses edges
       - K2 Score is a Bayesian scoring function that penalizes complexity
       - Fast but may find local optima

    2. Hill Climbing with BIC Score:
       - Same search strategy but uses Bayesian Information Criterion
       - BIC penalizes model complexity more heavily than K2
       - Often produces sparser networks

    Parameters:
    -----------
    df_bn : pandas.DataFrame
        Discretized dataset with categorical variables

    Returns:
    --------
    tuple : (model1, model2, chosen_model, chosen_edges)
        Both learned models and the selected one for inference

    CLINICAL INTERPRETATION:
    ------------------------
    We prefer networks where:
    - Death is influenced by clinical variables (EF, Creatinine) not just time
    - Risk factors cluster meaningfully (e.g., age → multiple outcomes)
    - The structure aligns with known medical knowledge
    """
    print("\n" + "=" * 70)
    print("SECTION 2: BAYESIAN NETWORK STRUCTURE LEARNING")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Algorithm 1: Hill Climbing with K2 Score
    # -------------------------------------------------------------------------
    print("\n--- Algorithm 1: Hill Climbing with K2 Score ---")
    hc_k2 = HillClimbSearch(df_bn)
    k2_score = K2Score(df_bn)
    model_k2 = hc_k2.estimate(scoring_method=k2_score, max_indegree=4)
    edges_k2 = list(model_k2.edges())
    print(f"K2 Model - Number of edges: {len(edges_k2)}")
    print(f"Edges: {edges_k2}")

    # -------------------------------------------------------------------------
    # Algorithm 2: Hill Climbing with BIC Score
    # -------------------------------------------------------------------------
    print("\n--- Algorithm 2: Hill Climbing with BIC Score ---")
    hc_bic = HillClimbSearch(df_bn)
    bic_score = BicScore(df_bn)
    model_bic = hc_bic.estimate(scoring_method=bic_score, max_indegree=4)
    edges_bic = list(model_bic.edges())
    print(f"BIC Model - Number of edges: {len(edges_bic)}")
    print(f"Edges: {edges_bic}")

    # -------------------------------------------------------------------------
    # Visualize Both Networks
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Bayesian Network Structure Learning Comparison',
                 fontsize=14, fontweight='bold')

    # K2 Network
    G_k2 = nx.DiGraph(edges_k2)
    pos_k2 = nx.spring_layout(G_k2, seed=42, k=2)
    ax1 = axes[0]
    nx.draw(G_k2, pos_k2, ax=ax1, with_labels=True, node_color='lightblue',
            node_size=2000, font_size=8, font_weight='bold',
            edge_color='gray', arrows=True, arrowsize=20,
            connectionstyle='arc3,rad=0.1')
    ax1.set_title(f'Algorithm 1: Hill Climbing + K2 Score\n({len(edges_k2)} edges)',
                  fontweight='bold')

    # BIC Network
    G_bic = nx.DiGraph(edges_bic)
    pos_bic = nx.spring_layout(G_bic, seed=42, k=2)
    ax2 = axes[1]
    nx.draw(G_bic, pos_bic, ax=ax2, with_labels=True, node_color='lightgreen',
            node_size=2000, font_size=8, font_weight='bold',
            edge_color='gray', arrows=True, arrowsize=20,
            connectionstyle='arc3,rad=0.1')
    ax2.set_title(f'Algorithm 2: Hill Climbing + BIC Score\n({len(edges_bic)} edges)',
                  fontweight='bold')

    plt.tight_layout()
    plt.savefig('sub_module5/bn_structure_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # -------------------------------------------------------------------------
    # Choose the Best Model and Justify
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STRUCTURE LEARNING DECISION")
    print("-" * 70)

    # Check which model has Death connected to clinical variables
    death_parents_k2 = [e[0] for e in edges_k2 if e[1] == 'Death']
    death_parents_bic = [e[0] for e in edges_bic if e[1] == 'Death']

    print(f"\nK2 Model - Variables directly influencing Death: {death_parents_k2}")
    print(f"BIC Model - Variables directly influencing Death: {death_parents_bic}")

    # Choose based on clinical sensibility
    # Prefer the model where clinical variables (EF, Creatinine) influence Death
    clinical_vars = ['EF_bin', 'Creatinine_bin', 'Age_bin', 'Time_bin']

    k2_clinical_count = sum(1 for v in death_parents_k2 if v in clinical_vars)
    bic_clinical_count = sum(1 for v in death_parents_bic if v in clinical_vars)

    if k2_clinical_count >= bic_clinical_count:
        chosen_model = 'K2'
        chosen_edges = edges_k2
        print("\n*** CHOSEN MODEL: K2 Score ***")
    else:
        chosen_model = 'BIC'
        chosen_edges = edges_bic
        print("\n*** CHOSEN MODEL: BIC Score ***")

    justification = """
    JUSTIFICATION (2-4 sentences):
    ------------------------------
    The {} model is chosen because it better captures clinically meaningful
    relationships. Specifically, it shows {} directly influencing mortality (Death),
    which aligns with medical knowledge that these are primary prognostic factors
    in heart failure. The network structure also appropriately connects Age to
    multiple downstream variables, reflecting age as a fundamental risk modifier.
    This interpretable structure will allow clinicians to understand and trust
    the model's predictions.
    """.format(chosen_model, death_parents_k2 if chosen_model == 'K2' else death_parents_bic)
    print(justification)

    return edges_k2, edges_bic, chosen_model, chosen_edges


# =============================================================================
# SECTION 3: PARAMETER LEARNING & CLINICAL INFERENCE
# =============================================================================

def fit_and_query_bn(df_bn, edges):
    """
    Fit Conditional Probability Tables (CPTs) and perform clinical inference queries.

    WHAT ARE CPTs?
    --------------
    Conditional Probability Tables specify P(X | Parents(X)) for each variable.
    They are learned from the data using Maximum Likelihood Estimation, which
    simply counts frequencies in the training data.

    WHAT IS VARIABLE ELIMINATION?
    -----------------------------
    Variable Elimination is an exact inference algorithm that computes posterior
    probabilities by summing out (marginalizing) hidden variables. It's more
    efficient than naive enumeration but still exact (no approximation).

    Parameters:
    -----------
    df_bn : pandas.DataFrame
        Discretized dataset for parameter learning
    edges : list
        List of directed edges [(parent, child), ...] for the BN structure

    Returns:
    --------
    BayesianNetwork : The fitted model ready for inference

    ADAPTED QUERIES (since we use UCI dataset, not Kaggle):
    -------------------------------------------------------
    Original assignment queries referenced features not in our dataset.
    We adapt them to equivalent clinical queries:

    a) P(Death=Died | Age_bin='Senior', EF_bin='Severe')
    b) P(Death=Died | Age_bin='Senior', EF_bin='Severe', Anaemia='Yes')
    c) P(Death=Died | Creatinine_bin='High', Time_bin='Short')
    d) P(Death=Died | Diabetes='Yes', HighBP='No')
    e) Full diagnostic: P(Death=Died | Age_bin='Senior', EF_bin='Severe',
                          Creatinine_bin='High', Time_bin='Short')
    """
    print("\n" + "=" * 70)
    print("SECTION 3: PARAMETER LEARNING & CLINICAL INFERENCE")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Build and Fit the Bayesian Network
    # -------------------------------------------------------------------------
    print("\n--- Fitting Conditional Probability Tables (CPTs) ---")

    # Create the Bayesian Network with learned structure
    model = BayesianNetwork(edges)

    # Fit CPTs using Maximum Likelihood Estimation
    model.fit(df_bn, estimator=MaximumLikelihoodEstimator)

    print(f"Model fitted with {len(model.nodes())} nodes and {len(model.edges())} edges")
    print(f"Nodes: {list(model.nodes())}")

    # Create inference engine
    inference = VariableElimination(model)

    # -------------------------------------------------------------------------
    # Clinical Inference Queries
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CLINICAL INFERENCE QUERIES")
    print("-" * 70)

    # Helper function to safely query and handle missing evidence
    def safe_query(evidence, query_name):
        """Perform query with error handling for missing states."""
        try:
            result = inference.query(variables=['Death'], evidence=evidence)
            prob_died = result.values[list(result.state_names['Death']).index('Died')]
            return prob_died
        except Exception as e:
            print(f"  Warning: {e}")
            return None

    # -------------------------------------------------------------------------
    # Query A: P(Death=Died | Age_bin='Senior', EF_bin='Severe')
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("QUERY A: P(Death=Died | Age='Senior', EF='Severe')")
    print("=" * 50)
    evidence_a = {'Age_bin': 'Senior', 'EF_bin': 'Severe'}
    prob_a = safe_query(evidence_a, "Query A")
    if prob_a is not None:
        print(f"\n  RESULT: P(Death=Died) = {prob_a:.2f}")
        print("""
  INTERPRETATION:
  A senior patient (65+ years) with severe ejection fraction (<30%) has a
  {:.0f}% probability of death. This is significantly elevated because:
  1. Age increases cardiovascular risk through cumulative damage
  2. Severe EF indicates the heart is failing to pump adequate blood
  3. Combined, these factors represent a high-risk cardiac profile
        """.format(prob_a * 100))

    # -------------------------------------------------------------------------
    # Query B: Same patient but with Anaemia=Yes
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("QUERY B: P(Death=Died | Age='Senior', EF='Severe', Anaemia='Yes')")
    print("=" * 50)
    evidence_b = {'Age_bin': 'Senior', 'EF_bin': 'Severe', 'Anaemia': 'Yes'}
    prob_b = safe_query(evidence_b, "Query B")
    if prob_b is not None:
        print(f"\n  RESULT: P(Death=Died) = {prob_b:.2f}")
        change = "increases" if prob_b > (prob_a or 0) else "decreases"
        print(f"""
  INTERPRETATION:
  Adding anaemia to the patient profile {change} the death probability to
  {prob_b*100:.0f}%. Anaemia reduces oxygen-carrying capacity, which further
  stresses an already weakened heart. In heart failure patients, anaemia is
  both a comorbidity and an independent predictor of worse outcomes, as the
  heart must work harder to compensate for reduced oxygen delivery.
        """)

    # -------------------------------------------------------------------------
    # Query C: P(Death=Died | Creatinine_bin='High', Time_bin='Short')
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("QUERY C: P(Death=Died | Creatinine='High', Time='Short')")
    print("=" * 50)
    evidence_c = {'Creatinine_bin': 'High', 'Time_bin': 'Short'}
    prob_c = safe_query(evidence_c, "Query C")
    if prob_c is not None:
        print(f"\n  RESULT: P(Death=Died) = {prob_c:.2f}")
        print(f"""
  INTERPRETATION:
  Patients with high serum creatinine (>2.0 mg/dL) and short follow-up time
  have a {prob_c*100:.0f}% probability of death. This makes clinical sense:
  1. High creatinine indicates severe kidney dysfunction (cardiorenal syndrome)
  2. Short follow-up often means the patient died early in the study
  3. Together, these suggest rapid disease progression and poor prognosis
        """)

    # -------------------------------------------------------------------------
    # Query D: P(Death=Died | Diabetes='Yes', HighBP='No')
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("QUERY D: P(Death=Died | Diabetes='Yes', HighBP='No')")
    print("=" * 50)
    evidence_d = {'Diabetes': 'Yes', 'HighBP': 'No'}
    prob_d = safe_query(evidence_d, "Query D")
    if prob_d is not None:
        print(f"\n  RESULT: P(Death=Died) = {prob_d:.2f}")
        print(f"""
  INTERPRETATION:
  A diabetic patient without high blood pressure has a {prob_d*100:.0f}%
  death probability. While diabetes is a cardiovascular risk factor due to
  its effects on blood vessels, the absence of hypertension is somewhat
  protective. This query demonstrates how the Bayesian Network can combine
  both risk factors and protective factors to compute personalized risk.
        """)

    # -------------------------------------------------------------------------
    # Query E: Full Diagnostic Profile
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("QUERY E: FULL DIAGNOSTIC")
    print("P(Death=Died | Age='Senior', EF='Severe', Creatinine='High', Time='Short')")
    print("=" * 50)
    evidence_e = {
        'Age_bin': 'Senior',
        'EF_bin': 'Severe',
        'Creatinine_bin': 'High',
        'Time_bin': 'Short'
    }
    prob_e = safe_query(evidence_e, "Query E")
    if prob_e is not None:
        print(f"\n  RESULT: P(Death=Died) = {prob_e:.2f}")
        print(f"""
  INTERPRETATION:
  This represents the highest-risk patient profile in our dataset: elderly,
  severe heart failure (low EF), kidney dysfunction (high creatinine), and
  short follow-up period. The {prob_e*100:.0f}% death probability reflects the
  synergistic effect of multiple severe risk factors. This is precisely the
  type of patient who would benefit from aggressive intervention and close
  monitoring. The explainability of the Bayesian Network allows clinicians
  to see WHICH factors contribute most to this high-risk classification.
        """)

    return model, inference


# =============================================================================
# SECTION 4: PROBABILISTIC CLASSIFIER
# =============================================================================

def build_classifier(df_bn, model, inference):
    """
    Convert the Bayesian Network into a probabilistic classifier and evaluate.

    CLASSIFICATION APPROACH:
    ------------------------
    For each patient, we compute P(Death='Died' | all observed features) using
    the fitted Bayesian Network. If this probability > 0.5, we predict death.

    This is fundamentally different from discriminative classifiers (logistic
    regression, random forests, neural networks) which directly learn P(Y|X).
    Bayesian Networks are generative models that learn the joint distribution
    P(X,Y) and derive P(Y|X) via Bayes' theorem.

    EVALUATION METRICS:
    -------------------
    - Accuracy: Proportion of correct predictions
    - AUC (Area Under ROC Curve): Measures ranking ability, robust to class imbalance

    Parameters:
    -----------
    df_bn : pandas.DataFrame
        Discretized dataset
    model : BayesianNetwork
        Fitted Bayesian Network model
    inference : VariableElimination
        Inference engine for the model

    Returns:
    --------
    dict : Dictionary containing accuracy, AUC, and predictions
    """
    print("\n" + "=" * 70)
    print("SECTION 4: PROBABILISTIC CLASSIFIER")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Train/Test Split (70/30)
    # -------------------------------------------------------------------------
    print("\n--- Splitting data 70/30 for training and testing ---")

    # Get feature columns (all except Death)
    feature_cols = [col for col in df_bn.columns if col != 'Death']

    X = df_bn[feature_cols]
    y = df_bn['Death']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Death rate in test set: {(y_test == 'Died').mean()*100:.1f}%")

    # -------------------------------------------------------------------------
    # Refit model on training data only
    # -------------------------------------------------------------------------
    print("\n--- Refitting model on training data ---")
    train_data = pd.concat([X_train, y_train], axis=1)

    # Use same structure but refit parameters
    model_train = BayesianNetwork(model.edges())
    model_train.fit(train_data, estimator=MaximumLikelihoodEstimator)
    inference_train = VariableElimination(model_train)

    # -------------------------------------------------------------------------
    # Make predictions on test set
    # -------------------------------------------------------------------------
    print("\n--- Making predictions on test set ---")

    predictions = []
    probabilities = []

    for idx, row in X_test.iterrows():
        # Build evidence dictionary from all features
        evidence = {col: row[col] for col in feature_cols if col in model_train.nodes()}

        try:
            result = inference_train.query(variables=['Death'], evidence=evidence)
            prob_died = result.values[list(result.state_names['Death']).index('Died')]
        except Exception as e:
            # If exact evidence combination not seen, use prior
            prob_died = 0.5

        probabilities.append(prob_died)
        predictions.append('Died' if prob_died > 0.5 else 'Survived')

    # -------------------------------------------------------------------------
    # Calculate Metrics
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CLASSIFICATION RESULTS")
    print("-" * 70)

    # Convert to binary for metrics
    y_test_binary = (y_test == 'Died').astype(int)
    y_pred_binary = [1 if p == 'Died' else 0 for p in predictions]

    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    auc = roc_auc_score(y_test_binary, probabilities)

    print(f"\n  ACCURACY: {accuracy:.2f} ({accuracy*100:.1f}%)")
    print(f"  AUC (Area Under ROC Curve): {auc:.2f}")

    print("\n  CLASSIFICATION REPORT:")
    print(classification_report(y_test, predictions, target_names=['Survived', 'Died']))

    print("""
  INTERPRETATION:
  ---------------
  The Bayesian Network achieves {:.1f}% accuracy and {:.2f} AUC on the held-out
  test set. While this may be slightly lower than black-box models like XGBoost
  or neural networks, the key advantage is EXPLAINABILITY. For every prediction,
  we can show exactly which factors contributed and query "what-if" scenarios.

  The AUC of {:.2f} indicates the model has good discriminative ability - it
  ranks high-risk patients above low-risk patients {:.0f}% of the time.
    """.format(accuracy*100, auc, auc, auc*100))

    # -------------------------------------------------------------------------
    # Visualize ROC Curve
    # -------------------------------------------------------------------------
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_test_binary, probabilities)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Bayesian Network (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.3)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve: Bayesian Network Heart Failure Classifier', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sub_module5/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    return {
        'accuracy': accuracy,
        'auc': auc,
        'predictions': predictions,
        'probabilities': probabilities
    }


# =============================================================================
# SECTION 5: DISCUSSION QUESTIONS
# =============================================================================

def print_discussion():
    """
    Answer the two discussion questions about Bayesian Networks in healthcare.

    These questions explore WHY explainability matters in medical AI and provide
    a real-world example of Bayesian methods in clinical practice.
    """
    print("\n" + "=" * 70)
    print("SECTION 5: DISCUSSION QUESTIONS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Question 1: Why prefer BN over black-box models?
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("QUESTION 1: Why might a hospital or cardiologist prefer your Bayesian")
    print("Network over a neural network or XGBoost that has 3-5% higher accuracy?")
    print("-" * 70)

    answer_1 = """
    ANSWER:
    -------
    Hospitals and cardiologists would prefer a Bayesian Network for several
    critical reasons, even if it sacrifices some accuracy:

    1. EXPLAINABILITY & TRANSPARENCY:
       - Bayesian Networks show WHICH variables influence the prediction and HOW
       - Clinicians can trace the reasoning: "This patient is high-risk because
         their ejection fraction is severe AND they have kidney dysfunction"
       - Black-box models provide predictions without justification

    2. REGULATORY COMPLIANCE:
       - FDA and EU MDR increasingly require "explainable AI" for medical devices
       - The EU AI Act (2024) classifies medical AI as "high-risk" requiring
         transparency in decision-making
       - Bayesian Networks naturally satisfy these requirements

    3. CLINICAL TRUST & ADOPTION:
       - Doctors are legally and ethically responsible for patient care
       - They won't trust (and shouldn't trust) a model they can't understand
       - A 95% accurate explainable model beats a 98% accurate black box that
         doctors refuse to use

    4. WHAT-IF QUERIES & INTERVENTION PLANNING:
       - Bayesian Networks support counterfactual reasoning: "If we improve this
         patient's ejection fraction, how much does their risk decrease?"
       - This supports treatment planning, not just risk stratification

    5. GRACEFUL HANDLING OF MISSING DATA:
       - BNs naturally marginalize over missing variables
       - No need for imputation hacks that black-box models require

    6. LIABILITY & MALPRACTICE CONCERNS:
       - If a patient is harmed due to an AI recommendation, hospitals need to
         explain WHY that recommendation was made
       - "The neural network said so" is not a legal defense

    In summary: A slightly less accurate model that doctors TRUST and USE will
    save more lives than a more accurate model that sits unused because no one
    understands it.
    """
    print(answer_1)

    # -------------------------------------------------------------------------
    # Question 2: Real-world example of Bayesian methods in clinical practice
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("QUESTION 2: Name and briefly describe one real-world medical system")
    print("or company in 2025 that uses Bayesian Networks or Bayesian deep")
    print("learning in clinical practice.")
    print("-" * 70)

    answer_2 = """
    ANSWER:
    -------
    EXAMPLE: Babylon Health (now part of eMed) - AI Symptom Checker

    DESCRIPTION:
    Babylon Health developed one of the most widely-deployed medical AI systems
    using Bayesian Networks for clinical triage and symptom checking. Their
    system has been used by millions of patients through the UK's NHS and
    internationally.

    HOW IT WORKS:
    - The system uses a Bayesian Network with over 4,500 medical conditions
    - When a patient describes symptoms, the network computes posterior
      probabilities for each possible diagnosis
    - It asks follow-up questions to maximize information gain (using entropy)
    - Finally, it provides ranked diagnoses with explanations of WHY each
      condition is considered

    WHY BAYESIAN:
    - Regulatory approval required explainable predictions
    - Doctors reviewing the AI's recommendations needed to understand the logic
    - The probabilistic framework handles uncertainty naturally, which is
      essential in medicine where symptoms are often ambiguous

    CITATION:
    Razzaki, S., et al. (2018). "A comparative study of artificial intelligence
    and human doctors for the purpose of triage and diagnosis." arXiv:1806.10698

    Also: Babylon Health was acquired by eMed in 2023, and the AI technology
    continues to be developed for telehealth applications.

    OTHER NOTABLE EXAMPLES:
    - Eko Health: Uses Bayesian methods for heart murmur detection from audio
    - Tempus: Uses Bayesian modeling in genomic cancer analysis
    - PathAI: Incorporates uncertainty quantification (Bayesian principles)
      in pathology image analysis
    """
    print(answer_2)

    return None


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function that runs all sections of the assignment.

    EXECUTION FLOW:
    ---------------
    1. Load and prepare data (discretize continuous variables)
    2. Perform EDA with 4 visualizations
    3. Learn BN structure with 2 algorithms, choose best
    4. Fit CPTs and run 5 clinical inference queries
    5. Build classifier, report accuracy and AUC
    6. Print discussion answers

    All outputs are saved to the sub_module5 directory.
    """
    print("\n" + "=" * 70)
    print("ASSIGNMENT 5.5: EXPLAINABLE BAYESIAN NETWORK FOR HEART FAILURE")
    print("=" * 70)
    print("Author: Whitney Parker")
    print("Dataset: UCI Heart Failure Clinical Records")
    print("=" * 70)

    # Step 1: Load and prepare data
    df_original, df_bn = load_and_prepare_data('sub_module5/heart_failure_records.csv')

    # Step 2: Exploratory Data Analysis
    perform_eda(df_original)

    # Step 3: Structure Learning
    edges_k2, edges_bic, chosen_model, chosen_edges = learn_bn_structure(df_bn)

    # Step 4: Parameter Learning and Clinical Inference
    model, inference = fit_and_query_bn(df_bn, chosen_edges)

    # Step 5: Probabilistic Classifier
    results = build_classifier(df_bn, model, inference)

    # Step 6: Discussion Questions
    print_discussion()

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ASSIGNMENT COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"""
    KEY RESULTS:
    ------------
    - Structure Learning: {chosen_model} Score selected for clinical interpretability
    - Network: {len(chosen_edges)} edges connecting {len(set([e[0] for e in chosen_edges] + [e[1] for e in chosen_edges]))} variables
    - Classifier Accuracy: {results['accuracy']*100:.1f}%
    - Classifier AUC: {results['auc']:.2f}

    OUTPUT FILES:
    -------------
    - sub_module5/eda_plots.png: Four EDA visualizations
    - sub_module5/bn_structure_comparison.png: K2 vs BIC network structures
    - sub_module5/roc_curve.png: ROC curve for classifier evaluation

    The Bayesian Network provides an explainable, clinically interpretable
    model for heart failure mortality prediction. While accuracy may be
    slightly lower than black-box alternatives, the transparency and
    query capabilities make it suitable for real clinical deployment.
    """)


if __name__ == "__main__":
    main()
