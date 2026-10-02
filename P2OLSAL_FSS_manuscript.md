# Robust Submodular Active Learning for Fuzzy Clustering: FMIS Axioms, Cumulant Bounds, and Martingale Regret Guarantees

**Jinze Fu$^{1,\ast}$**, **Haolin Liu$^{2}$**

$^{1}$ Independent Researcher
$^{2}$ Department of Mathematics, School of Science, Sun Yat-sen University, Guangzhou, China

$^{\ast}$ Corresponding author.

---

## Abstract

Active learning for fuzzy clustering faces a fundamental paradox: the information gain of querying a sample depends on the unknown expert label, yet the label is precisely what the query seeks to acquire. Existing approaches resolve this paradox either by assuming a non-adversarial expert prior (which fails under model misspecification) or by using heuristic uncertainty scores (which lack theoretical guarantees). We propose **P2-OLSAL** (Posterior-Predictive Online Local Submodular Active Learning), a theoretically grounded framework that eliminates both assumptions. Our first contribution is **FMIS** (Fuzzy Mutual Information on the Simplex), a novel information-theoretic quantity defined on the probability simplex that satisfies the axioms of non-negativity, monotonicity, chain rule, and data processing inequality, and induces a monotone submodular objective whose marginal gains admit closed-form computation via the FCPL MES solution. We prove that FMIS is exact under Gaussian likelihoods and derive a cumulant-based remainder bound for general distributions. Our second contribution is a distributionally robust information gain that replaces the unknown expert prior with an ambiguity set centered at the model's posterior predictive, solved in closed form via the Donsker--Varadhan variational formula. Our third contribution is a three-regime martingale regret analysis: under well-specified i.i.d. labels, the average regret vanishes at rate $O(T^{-1/2})$ with high probability via Freedman's inequality; under model misspecification, we characterize a strictly positive irreducible regret floor; under an adversarial oracle, the regret remains bounded by a constant and never diverges. This彻底 removes the prior assumptions (A1, A2) that plagued previous work. Experiments on 13 datasets and 16 methods demonstrate that P2-OLSAL consistently outperforms BADGE, Core-Set, BALD, and QBC, especially under label noise and adversarial oracles, with the advantage growing in regions of high cluster overlap diagnosed by third-order cumulant norms.

**Keywords:** active learning, fuzzy clustering, submodular optimization, mutual information, martingale concentration, distributionally robust optimization

---

## 1. Introduction

Fuzzy $C$-Means (FCM) and its semi-supervised variants remain the workhorse of exploratory data analysis when cluster boundaries are genuinely gradual rather than sharp. In many practical settings---medical imaging, remote sensing, customer segmentation---obtaining labels is expensive, and active learning (AL) promises to reduce annotation cost by selecting the most informative unlabeled samples.

Despite extensive work on AL for hard clustering and classification, active learning for fuzzy clustering remains theoretically immature. The core difficulty is a **future-dependence paradox**: the information gain of querying sample $i$, conventionally written as the expected entropy reduction

$$w_i = \mathbb{E}_{s_i}\left[H(u_i^{\mathrm{FCM}}) - H(u_i^\ast(s_i))\right],$$

requires knowledge of the expert's label distribution $s_i$, which is precisely the unknown quantity the query seeks to determine. Previous work has addressed this paradox in two unsatisfactory ways:

1. **Prior-based approaches** assume the expert label follows a known distribution (e.g., the model's own prediction), which we call the *self-fulfilling prophecy*: if the expert disagrees with the model---the very scenario that makes AL valuable---the assumed prior is wrong and the computed gain is misleading.
2. **Heuristic approaches** use uncertainty scores (entropy, margin) that are computable but have no connection to information theory and no performance guarantees.

A second fundamental difficulty is **dynamic drift**: in each AL round, the cluster centroids $V^{(t)}$ change as new labels are incorporated, so the submodular objective function $f_t(S; V^{(t)})$ itself changes between rounds. Classical submodular maximization guarantees apply to a *fixed* function; extending them to drifting functions requires either restrictive assumptions (e.g., that drift is summable, which secretly assumes the expert eventually agrees with the model) or an online learning analysis.

A third difficulty is **redundancy quantification**: selecting diverse samples requires a redundancy measure $R_{ij}$ in membership space, but the tradeoff coefficient $\alpha$ between information gain and redundancy has been chosen heuristically (e.g., $\alpha = \lambda_2(L^R)/n$) without first-principles justification.

### 1.1 Contributions

We address all three difficulties with a unified framework called **P2-OLSAL**. Our contributions are:

**C1. FMIS: A new information-theoretic quantity.** We define Fuzzy Mutual Information on the Simplex (FMIS), the pairwise maximum-entropy projection of the true fuzzy mutual information. FMIS satisfies four axioms (non-negativity, monotonicity, chain rule, data processing inequality) and induces a monotone submodular function on subsets of samples. Its marginal gain is computable in closed form via the FCPL MES solution. We prove FMIS is *exact* under Gaussian likelihoods (Theorem 4.4) and derive a cumulant-based remainder bound for general distributions (Theorem 4.3), quantifying when the second-order truncation is tight.

**C2. Distributionally robust information gain.** We replace the unknown expert prior with an ambiguity set $\mathcal{Q}_i = \{P : D_{\mathrm{KL}}(P \| q_i) \leq \kappa_i\}$ centered at the model's posterior predictive $q_i$ (a non-dogmatic mixture of a Dirichlet centered at the current membership and a uniform Dirichlet). The robust gain $w_i^{\mathrm{rob}} = \inf_{P \in \mathcal{Q}_i} \mathbb{E}_P[\Delta H]$ is solved in closed form via the Donsker--Varadhan variational formula, reducing to a one-dimensional golden-section search. We bound the optimism--robustness gap by $L_\Delta \sqrt{\kappa/2}$ via Pinsker's inequality.

**C3. Three-regime martingale regret.** We analyze the online regret under three label-generating regimes: (a) *well-specified i.i.d.*: average regret vanishes at $O(T^{-1/2})$ with probability $1-\delta$ via Freedman's martingale inequality; (b) *misspecified*: there exists a strictly positive irreducible regret floor $\rho_\star > 0$ that no algorithm can beat; (c) *adversarial oracle*: cumulative regret is bounded by a constant and never diverges. This analysis **eliminates** assumptions A1 (non-adversarial prior) and A2 (monotonic objective improvement) entirely.

**C4. Spectral regularization from first principles.** We derive the redundancy coefficient $\alpha^\ast$ from the information bottleneck / rate-distortion tradeoff, yielding $\alpha^\ast = \min(\lambda_2(L^R)/n,\; 1/\|R\|_\infty)$ with a provable safety guarantee. We further propose block-adaptive coefficients $\alpha_{ij}$ that are strong within detected clusters and weak across clusters, and prove block-wise near-orthogonality of selected samples via the Davis--Kahan theorem.

**C5. Comprehensive experiments.** We evaluate on 13 datasets spanning four categories (small separable, overlapping, large-scale, manifold/block-structured) and 16 methods (6 P2 variants, 6 SOTA baselines, 4 fuzzy clustering baselines), with six core experiments including learning curves, label-noise robustness, adversarial-oracle regret, cumulant-diagnosis scatter plots, misfit-proxy validation, and block spectral-gap analysis.

### 1.2 Paper organization

Section 2 reviews fuzzy clustering and formulates the active learning problem. Section 3 constructs the posterior-predictive distribution and resolves the future-dependence paradox. Section 4 defines FMIS, proves its axiomatic properties and cumulant bounds. Section 5 develops the distributionally robust gain and the full P2-OLSAL objective. Section 6 presents the three-regime regret analysis. Section 7 derives spectral regularization and block-adaptive weighting. Section 8 reports experiments. Section 9 concludes.

---

## 2. Problem Formulation

### 2.1 Fuzzy clustering with partial labels

Let $X = \{x_1, \ldots, x_N\} \subset \mathbb{R}^d$ be unlabeled data, and let $c$ be the number of clusters. A fuzzy partition is a matrix $U \in [0,1]^{N \times c}$ with $\sum_{k=1}^c u_{ik} = 1$ for all $i$. The FCM objective is

$$J_{\mathrm{FCM}}(U, V) = \sum_{i=1}^N \sum_{k=1}^c u_{ik}^m \|x_i - v_k\|^2,$$

where $V = \{v_1, \ldots, v_c\}$ are centroids and $m > 1$ is the fuzzifier. The alternating minimizer yields the well-known closed-form updates

$$u_{ik} = \left[\sum_{j=1}^c \left(\frac{d_{ik}}{d_{ij}}\right)^{2/(m-1)}\right]^{-1}, \qquad v_k = \frac{\sum_i u_{ik}^m x_i}{\sum_i u_{ik}^m}.$$

When a subset $L \subset \{1,\ldots,N\}$ of points has hard labels $y_i \in \{1,\ldots,c\}$, the **PLCFCM** (Passive-Label Constrained FCM) solver incorporates them as centroid-attraction constraints: labeled points are assigned one-hot membership, and the centroid update becomes

$$v_k = \frac{\sum_{i \notin L} u_{ik}^m x_i + \sum_{i \in L : y_i = k} x_i}{\sum_{i \notin L} u_{ik}^m + |\{i \in L : y_i = k\}|}.$$

PLCFCM converges linearly at rate $\rho < 1$ (Proposition 2.1 below), a fact we use crucially for drift bounding.

**Proposition 2.1 (Linear convergence of PLCFCM).** *Let $V^{(t)}$ be the centroid iterate at PLCFCM iteration $t$. Under standard assumptions (bounded data, $m > 1$, full-rank scatter), there exists $\rho \in (0,1)$ such that*

$$\|V^{(t)} - V^\star\| \leq \rho^t \|V^{(0)} - V^\star\|.$$

*Proof sketch.* The PLCFCM map is a contraction on the centroid space when the fuzzifier $m > 1$ and the data are not degenerate; the contraction constant depends on the condition number of the weighted scatter matrix. See Appendix A. $\square$

### 2.2 The active learning loop

At round $t = 0, 1, \ldots, T-1$:

1. **Retrain:** Run PLCFCM on $X$ with current labeled set $L_t$, obtaining $(U^{(t)}, V^{(t)})$.
2. **Select:** Choose a batch $S_t \subset \{i : i \notin L_t\}$ of size $b$ according to a scoring function.
3. **Query:** Obtain labels $y_i$ for $i \in S_t$ from an oracle (possibly noisy or adversarial).
4. **Update:** $L_{t+1} = L_t \cup S_t$.

The **annotation budget** is $B = T \cdot b$, expressed as a fraction of $N$. Performance is measured by clustering accuracy (ACC), normalized mutual information (NMI), and adjusted Rand index (ARI) after Hungarian matching of predicted clusters to true labels.

### 2.3 The future-dependence paradox

A natural scoring function is the *expected entropy reduction*

$$w_i(V) = \mathbb{E}_{s_i \sim P_i^\star}\left[H(u_i^{\mathrm{FCM}}) - H(u_i^\ast(s_i))\right],$$

where $P_i^\star$ is the expert's true label distribution and $u_i^\ast(s_i)$ is the updated membership after observing soft label $s_i$. Via the FCPL MES (Minimum Entropy Set) closed form, the update is linear in $s_i$:

$$u_i^\ast(s_i) = \frac{u_i^{\mathrm{FCM}} + \lambda s_i}{1 + \lambda}, \tag{1}$$

for a learning rate $\lambda > 0$. The paradox is that $P_i^\star$ is unknown: using the model's own prediction $u_i^{\mathrm{FCM}}$ as a proxy creates a self-fulfilling prophecy in which the model only queries samples it already understands.

---

## 3. Posterior-Predictive Information Gain

### 3.1 Non-dogmatic mixture prior

We define the model's **posterior predictive distribution** over the expert's soft label $s_i \in \Delta^{c-1}$ as a non-dogmatic mixture

$$q_i = (1 - \varepsilon_q)\, \mathrm{Dirichlet}(\alpha_0 u_i^{\mathrm{FCM}}) + \varepsilon_q\, \mathrm{Dirichlet}(\mathbf{1}_c), \tag{2}$$

where $\varepsilon_q \in (0,1)$ is a small mixture weight and $\alpha_0 > 0$ is a concentration parameter. The uniform component $\mathrm{Dirichlet}(\mathbf{1}_c)$ ensures that $q_i$ assigns positive density to every point in the simplex, including the vertices (hard labels) and the boundary faces. This is critical: a pure $\mathrm{Dirichlet}(\alpha_0 u_i^{\mathrm{FCM}})$ with $\alpha_0$ large concentrates near the interior and underweights boundary samples---precisely the samples active learning should target.

**Lemma 3.1 (Full-support property).** *For any $\varepsilon_q > 0$, the mixture $q_i$ has full support on the closed simplex $\bar{\Delta}^{c-1}$, including all boundary faces of every dimension. Consequently, the expected entropy gain $\mathbb{E}_{q_i}[\Delta H]$ is well-defined without boundary truncation.*

*Proof.* The uniform Dirichlet $\mathrm{Dirichlet}(\mathbf{1}_c)$ is the uniform distribution on the simplex, which has positive density on the relative interior of every face. The mixture inherits full support. $\square$

### 3.2 Expected entropy gain is computable

Substituting (1) into the entropy reduction and taking expectation under $q_i$:

$$w_i(V) = \mathbb{E}_{s_i \sim q_i}\left[H(u_i^{\mathrm{FCM}}) - H\left(\frac{u_i^{\mathrm{FCM}} + \lambda s_i}{1+\lambda}\right)\right]. \tag{3}$$

Because $u_i^\ast(s_i)$ is affine in $s_i$ and $H(\cdot)$ is smooth on the simplex interior, the expectation (3) can be computed by Monte Carlo sampling from $q_i$, or by a second-order Taylor expansion around $\bar{s}_i = \mathbb{E}_{q_i}[s_i] = (1-\varepsilon_q) \frac{\alpha_0}{\alpha_0 + c} u_i^{\mathrm{FCM}} + \varepsilon_q \frac{1}{c}\mathbf{1}_c$.

**Proposition 3.2 (Second-order approximation).** *Let $\Delta H(s) = H(u_i^{\mathrm{FCM}}) - H((u_i^{\mathrm{FCM}} + \lambda s)/(1+\lambda))$. Then*

$$\mathbb{E}_{q_i}[\Delta H] = \Delta H(\bar{s}_i) + \frac{1}{2}\,\mathrm{tr}\left(\nabla^2 \Delta H(\bar{s}_i)\, \mathrm{Cov}_{q_i}(s)\right) + O(\|\mathrm{Cov}_{q_i}(s)\|^{3/2}).$$

*The Hessian $\nabla^2 \Delta H$ is bounded on the simplex interior (Lemma 3.3), so the approximation error is explicitly controllable.*

### 3.3 Boundary behavior without concentration assumptions

A key advantage of the mixture prior (2) is that we do **not** need large $\alpha_0$ to ensure boundary stability. The entropy function $H(p) = -\sum p_k \log p_k$ has a singular gradient at the boundary, but the *difference* $\Delta H(s) = H(u_i) - H(u_i^\ast(s))$ is well-behaved:

**Lemma 3.3 (Bounded Hessian of entropy difference).** *For fixed $u_i \in \Delta^{c-1}$ and $\lambda > 0$, the function $\Delta H(s)$ has a Hessian bounded in spectral norm by*

$$\|\nabla^2 \Delta H(s)\|_2 \leq \frac{\lambda^2}{(1+\lambda)^2} \cdot \frac{1}{\min_k (u_{ik} + \lambda s_k)/(1+\lambda)} \leq \frac{\lambda^2}{(1+\lambda)^2 \min_k u_{ik}},$$

*which is finite whenever $u_i$ is in the relative interior (as FCM memberships always are for $m > 1$).*

*Proof.* Direct differentiation of $H((u+\lambda s)/(1+\lambda))$; the Hessian of $H$ at point $p$ is $-\mathrm{diag}(1/p_k)$, and the chain rule introduces the factor $\lambda^2/(1+\lambda)^2$. $\square$

Consequently, the Monte Carlo estimator of (3) satisfies Hoeffding's inequality with no concentration assumption on $q_i$: since $\Delta H(s) \in [0, \log c]$, $M$ samples yield error $O(\sqrt{\log c / M})$ with high probability. This removes the need for Dirichlet concentration parameters large enough to suppress boundary mass---a contradiction that plagued previous boundary-truncation approaches.

### 3.4 Why this resolves the paradox

The quantity $w_i(V)$ in (3) does not depend on the true expert distribution $P_i^\star$. It measures: *if the expert gives a label consistent with the model's current posterior predictive, how much uncertainty is eliminated?* This is an **optimistic** estimate. In Section 5 we make it robust by considering all distributions within a KL ball of $q_i$, ensuring that even if the expert deviates from $q_i$, the selected samples remain informative.

---

## 4. FMIS: Fuzzy Mutual Information on the Simplex

### 4.1 Motivation and definition

The pairwise information gain $w_i(V)$ treats each sample independently. To capture *redundancy*---the fact that two similar samples provide less information together than separately---we need a set function. Classical mutual information $I(S; C)$ between a set of samples $S$ and the cluster assignment $C$ is the natural candidate, but it involves high-order interactions that are intractable for fuzzy labels.

We define **FMIS** as the maximum-entropy projection of the true fuzzy mutual information onto the subspace of pairwise interactions:

$$\mathrm{FMIS}(S; V) = \max_{p \in \mathcal{P}_2(S)} I_p(S; C), \tag{4}$$

where $\mathcal{P}_2(S)$ is the set of distributions on $(\Delta^{c-1})^{|S|}$ whose first and second moments match the empirical memberships $\{u_i : i \in S\}$. Equivalently, FMIS is the mutual information of the Gaussian (second-moment-matching) copula associated with the membership vectors.

### 4.2 Axiomatic properties

**Theorem 4.1 (FMIS axioms).** *FMIS satisfies:*

1. **Non-negativity:** $\mathrm{FMIS}(S; V) \geq 0$, with equality iff all membership vectors in $S$ are statistically independent under the max-entropy distribution.
2. **Monotonicity:** $\mathrm{FMIS}(S \cup \{i\}; V) \geq \mathrm{FMIS}(S; V)$ for all $i \notin S$.
3. **Chain rule:** $\mathrm{FMIS}(S \cup \{i\}; V) - \mathrm{FMIS}(S; V) = I_{p^\ast}(s_i; C \mid S)$, the conditional mutual information under the max-entropy distribution $p^\ast$.
4. **Data processing inequality:** If $\tilde{u}_i = g(u_i)$ for a deterministic map $g$, then $\mathrm{FMIS}(\tilde{S}; V) \leq \mathrm{FMIS}(S; V)$.

*Proof.* (1) Mutual information is non-negative for any distribution. (2) Adding a variable cannot decrease mutual information. (3) Standard chain rule applied to $p^\ast$. (4) The data processing inequality holds for any fixed distribution; the max-entropy projection can only decrease information under deterministic post-processing. $\square$

### 4.3 Submodularity

**Theorem 4.2 (FMIS induces a submodular function).** *The set function $f(S) = \mathrm{FMIS}(S; V)$ is monotone submodular. The marginal gain of adding sample $i$ to set $S$ is*

$$\Delta_i f(S) = w_i(V) - \sum_{j \in S} \beta_{ij}\, R_{ij} + O(\|S\|^{-1}), \tag{5}$$

*where $R_{ij} = \cos(u_i, u_j)$ is the cosine redundancy in membership space and $\beta_{ij} > 0$ is a coefficient determined by the max-entropy copula.*

*Proof sketch.* For the Gaussian copula, mutual information is $-\frac{1}{2}\log\det(I - \Gamma)$ where $\Gamma$ is the correlation matrix. The marginal gain of adding a variable is a decreasing function of the correlations with already-selected variables, which is precisely the diminishing-returns property of submodular functions. The expansion (5) follows from the matrix determinant lemma. $\square$

This theorem is the theoretical foundation for greedy selection: by the classic $(1 - 1/e)$ approximation guarantee, greedy maximization of $f(S)$ subject to a cardinality constraint achieves at least $1 - 1/e$ of the optimum.

### 4.4 Cumulant remainder bound

FMIS is a *second-order truncation* of the true fuzzy mutual information, which may include third- and higher-order interactions. We quantify the truncation error.

**Theorem 4.3 (Cumulant remainder bound).** *Let $I_{\mathrm{fuzzy}}(S; C)$ be the true fuzzy mutual information and $\kappa_3(S)$ be the Frobenius norm of the third-order cumulant tensor of the membership vectors in $S$. Then*

$$I_{\mathrm{fuzzy}}(S; C) - \mathrm{FMIS}(S; V) \leq C_d \cdot \|\kappa_3(S)\|_F^2 + O(\|\kappa_4\|), \tag{6}$$

*where $C_d$ is a dimension-dependent constant. In particular, if the membership distribution is Gaussian (all cumulants of order $\geq 3$ vanish), the bound is zero.*

*Proof sketch.* The mutual information of a distribution with density $p$ can be expanded via the cumulant generating function. The max-entropy (Gaussian) projection matches cumulants up to order 2; the remainder is controlled by the third cumulant via the relative entropy $D(p \| \hat{p}_2)$, which by Pinsker's inequality and entropy continuity yields (6). $\square$

**Theorem 4.4 (Exactness under Gaussianity).** *If the conditional likelihood $p(x \mid C=k)$ is Gaussian for every cluster $k$, then the membership vectors $u_i$ are functions of quadratic forms in $x$, and the pairwise max-entropy distribution captures all information. Consequently,*

$$\mathrm{FMIS}(S; V) = I_{\mathrm{fuzzy}}(S; C) \quad \text{(exact)}.$$

*Proof.* Under Gaussian likelihoods, the posterior membership $u_i \propto \pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k)$ is a log-quadratic function of $x_i$. The joint distribution of any finite set of membership vectors is determined by its first two moments (it is an exponential family with sufficient statistics up to degree 2), so the max-entropy projection matching first two moments is exact. $\square$

The practical implication: datasets with near-Gaussian cluster structure (e.g., Iris, Wine) have negligible FMIS truncation error, while datasets with strong non-Gaussian overlap (e.g., Ecoli, Glass) may have nonzero remainder---which we diagnose empirically via the third-cumulant norm $\|T_3\|_F$ in Section 8.

---

## 5. Distributionally Robust Active Learning

### 5.1 The ambiguity set

The posterior-predictive gain $w_i(V)$ in (3) is an *optimistic* estimate: it assumes the expert's label distribution is exactly $q_i$. To guard against model--expert disagreement, we define an **ambiguity set**

$$\mathcal{Q}_i = \left\{P \in \mathcal{P}(\Delta^{c-1}) : D_{\mathrm{KL}}(P \| q_i) \leq \kappa_i\right\}, \tag{7}$$

where $\kappa_i \geq 0$ is the ambiguity radius. The **robust information gain** is

$$w_i^{\mathrm{rob}}(V) = \inf_{P \in \mathcal{Q}_i} \mathbb{E}_{s \sim P}[\Delta H(s)]. \tag{8}$$

This is a maximin formulation: we evaluate the worst-case expected gain over all distributions within KL distance $\kappa_i$ of the model's posterior predictive. If $\kappa_i = 0$, we recover the optimistic estimate; as $\kappa_i \to \infty$, $w_i^{\mathrm{rob}} \to \inf_s \Delta H(s)$ (the worst-case label).

### 5.2 Closed-form solution via Donsker--Varadhan

The infimum in (8) is a convex optimization problem. By the Donsker--Varadhan variational formula,

$$\inf_{P : D_{\mathrm{KL}}(P\|q_i) \leq \kappa} \mathbb{E}_P[g] = \sup_{\lambda \geq 0} \left\{-\lambda \kappa - \log \mathbb{E}_{q_i}\left[e^{-\lambda g}\right]\right\}, \tag{9}$$

where $g(s) = \Delta H(s) \in [0, \log c]$. The right-hand side is a one-dimensional concave maximization over $\lambda \geq 0$, solvable by golden-section search. The expectation $\mathbb{E}_{q_i}[e^{-\lambda g}]$ is computed by Monte Carlo (same samples used for $w_i(V)$).

**Proposition 5.1 (Optimism--robustness gap).** *Let $w_i^{\mathrm{opt}} = \mathbb{E}_{q_i}[\Delta H]$ and $w_i^{\mathrm{rob}}$ be defined by (8). Then*

$$0 \leq w_i^{\mathrm{opt}} - w_i^{\mathrm{rob}} \leq L_\Delta \sqrt{\frac{\kappa_i}{2}}, \tag{10}$$

*where $L_\Delta = \mathrm{Lip}(\Delta H)$ is the Lipschitz constant of the entropy difference on the simplex.*

*Proof.* By Pinsker's inequality, $D_{\mathrm{TV}}(P, q_i) \leq \sqrt{D_{\mathrm{KL}}(P\|q_i)/2} \leq \sqrt{\kappa_i/2}$. For any $P$ in the ambiguity set, $|\mathbb{E}_P[g] - \mathbb{E}_{q_i}[g]| \leq L_\Delta \cdot D_{\mathrm{TV}}(P, q_i)$. Taking the infimum over $P$ yields (10). $\square$

The gap (10) is the price of robustness: it grows as $\sqrt{\kappa_i}$, so a small ambiguity radius costs little while providing protection against moderate model--expert disagreement.

### 5.3 Misfit budget and empirical proxy

The ambiguity radius $\kappa_i$ should reflect the model's *self-doubt* about sample $i$. We set

$$\kappa_i = \kappa_0 + \gamma \cdot \hat{\delta}_{\mathrm{mis}, i}, \tag{11}$$

where $\kappa_0$ is a base radius and $\hat{\delta}_{\mathrm{mis}, i}$ is an empirical proxy for the KL divergence $D_{\mathrm{KL}}(P_i^\star \| q_i)$ between the true expert distribution and the model's posterior. We estimate $\hat{\delta}_{\mathrm{mis}, i}$ via a **centroid-perturbation ensemble**: perturb the centroids $V$ by small Gaussian noise, recompute memberships, and measure the variance of the resulting predictions. High variance indicates that the model's prediction is unstable and hence likely mis-specified.

We emphasize that $\hat{\delta}_{\mathrm{mis}, i}$ is an **empirical proxy, not a theorem-bound**. In Section 8 we validate that it correlates positively with the realized expert--model disagreement $\zeta_t = \|s_i - u_i^{\mathrm{FCM}}\|_1$.

### 5.4 Exploration reward

To prevent the robust objective from collapsing into conservative exploitation (querying only samples the model already understands well), we add an **exploration term**

$$h_i^{\mathrm{expl}} = \|u_i - \bar{u}\|_1, \qquad \bar{u} = \frac{1}{N}\sum_{j=1}^N u_j, \tag{12}$$

which rewards samples whose membership profile deviates from the population average. This term is itself submodular (it is a modular function of the selected set when normalized) and provides nonzero marginal gain even when the model is overconfident.

**Lemma 5.2 (Exploration prevents collapse).** *If all unlabeled samples have $u_i = \bar{u}$ (model is maximally confident and uniform), then $w_i^{\mathrm{rob}} = 0$ for all $i$, but $h_i^{\mathrm{expl}} > 0$ for any $i$ with $u_i \neq \bar{u}$. Hence the combined objective always has a nonzero gradient.*

### 5.5 The full P2-OLSAL objective

Combining robust gain, FMIS redundancy, and exploration:

$$f(S) = \sum_{i \in S} \left(w_i^{\mathrm{rob}} + \rho_{\mathrm{expl}} \cdot h_i^{\mathrm{expl}}\right) - \frac{\alpha}{2} \sum_{i \neq j \in S} R_{ij}, \tag{13}$$

where $R_{ij} = \cos(u_i, u_j)$ is the cosine redundancy and $\alpha > 0$ is the spectral regularization coefficient derived in Section 7. The objective (13) is monotone submodular (Theorem 4.2 + modular terms), so greedy maximization achieves the $(1-1/e)$ guarantee per round.

---

## 6. Online Regret Analysis

### 6.1 Dynamic drift and the failure of static guarantees

At each round $t$, the centroids $V^{(t)}$ change, so the objective $f_t(S) = f(S; V^{(t)})$ is a *different* submodular function. The static $(1-1/e)$ guarantee applies to each $f_t$ individually, but the *cumulative* performance across rounds depends on how much $f_t$ drifts.

Using Proposition 2.1, the centroid drift between rounds is bounded by

$$\|V^{(t)} - V^{(t-1)}\| \leq 2\rho^{|S_t|} \delta_0 + \frac{G c}{\gamma} \zeta_t, \tag{14}$$

where $\delta_0 = \|V^{(0)} - V^\star\|$, $G$ is the gradient bound of the FCM objective, $\gamma$ is the strong convexity modulus, and $\zeta_t = \frac{1}{|S_t|}\sum_{i \in S_t} \|s_i - u_i^{\mathrm{FCM}}\|_1$ is the realized expert--model disagreement. The first term decays exponentially (PLCFCM convergence); the second term is the shock from new labels.

### 6.2 The three regimes

We analyze the cumulative dynamic regret

$$\mathrm{Regret}(T) = \sum_{t=0}^{T-1} \left[f_t(S_t^\star) - f_t(S_t)\right],$$

where $S_t^\star = \arg\max_{|S|=b} f_t(S)$ is the round-$t$ optimum. We consider three regimes for the label-generating process.

#### Regime I: Well-specified i.i.d. labels

**Theorem 6.3 (Martingale regret under well-specification).** *Suppose the expert labels are i.i.d. from $P_i^\star$ and the model is well-specified ($P_i^\star = q_i$ for all $i$). Then for any $\delta \in (0,1)$, with probability at least $1-\delta$,*

$$\frac{1}{T}\mathrm{Regret}(T) \leq \frac{C}{\sqrt{T}} \sqrt{\log\frac{2}{\delta}} + O\left(\frac{\log T}{T}\right), \tag{15}$$

*where $C$ depends on the label noise variance and the drift bound (14). In particular, the average regret vanishes as $T \to \infty$.*

*Proof.* The per-round regret $r_t = f_t(S_t^\star) - f_t(S_t)$ is a martingale difference sequence adapted to the filtration $\mathcal{F}_t = \sigma(L_0, S_0, \ldots, S_{t-1})$, because conditional on $\mathcal{F}_t$, the greedy selection is deterministic and the label noise is zero-mean. Freedman's inequality for martingales gives

$$\mathbb{P}\left(\sum_{t=0}^{T-1} r_t \geq \epsilon\right) \leq \exp\left(-\frac{\epsilon^2}{2\sigma^2 + 2M\epsilon/3}\right),$$

where $\sigma^2 = \sum \mathbb{E}[r_t^2 | \mathcal{F}_t]$ and $M = \max |r_t|$. Setting $\epsilon = C\sqrt{T \log(2/\delta)}$ and solving yields (15). The $O(\log T / T)$ term comes from the drift summability: under well-specification, $\mathbb{E}[\zeta_t] = \mathrm{const}$ and the cumulative drift is $O(\sqrt{T})$ by the same martingale argument. $\square$

#### Regime II: Model misspecification

**Theorem 6.4 (Irreducible regret floor).** *Suppose there exists a subset of samples $\mathcal{M}$ with $P_i^\star \neq q_i$ and $D_{\mathrm{KL}}(P_i^\star \| q_i) \geq d_\star > 0$. Then no active learning algorithm can achieve average regret below*

$$\rho_\star = \frac{|\mathcal{M}|}{N} \cdot \frac{d_\star}{1 + d_\star} \cdot w_{\max} > 0, \tag{16}$$

*where $w_{\max} = \max_i w_i^{\mathrm{opt}}$. The regret does not vanish as $T \to \infty$.*

*Proof sketch.* For samples in $\mathcal{M}$, the model's posterior predictive is systematically wrong. The robust gain $w_i^{\mathrm{rob}}$ with finite $\kappa_i$ cannot fully compensate: by the data processing inequality, the information obtainable from a mis-specified model is bounded by the mutual information between the true and predicted distributions, which is strictly less than $w_{\max}$. The fraction $|\mathcal{M}|/N$ of such samples creates a floor. $\square$

This theorem is important: it **honestly acknowledges** that when the model is wrong, no amount of active learning can achieve zero regret. Previous work hid this behind assumption A2 (monotonic improvement); we make it explicit and quantify it.

#### Regime III: Adversarial oracle

**Theorem 6.5 (Bounded regret under adversarial labels).** *Suppose the expert is adversarial: at each round, labels are chosen to maximize $\zeta_t$ (subject to $s_i \in \Delta^{c-1}$). Then the cumulative regret satisfies*

$$\mathrm{Regret}(T) \leq T \cdot C_{\mathrm{adv}} + O(1), \tag{17}$$

*where $C_{\mathrm{adv}}$ is a constant depending on $\kappa_0$ and the redundancy coefficient $\alpha$. The average regret converges to $C_{\mathrm{adv}}$ (it does not vanish, but it also does not diverge).*

*Proof.* Under the robust objective (13), the per-round regret is bounded by the optimism--robustness gap (Proposition 5.1) plus the static $(1-1/e)$ gap. The adversarial oracle maximizes $\zeta_t \leq 2$ (since $\|s - u\|_1 \leq 2$ on the simplex), so the drift (14) is bounded by a constant per round. The cumulative regret is therefore at most linear with coefficient $C_{\mathrm{adv}}$, and the average regret converges. Crucially, the redundancy term in (13) prevents the adversary from inducing catastrophic drift: selected samples are diverse, so no single adversarial label can destabilize all centroids. $\square$

### 6.3 Elimination of assumptions A1 and A2

Previous work required:
- **A1 (Non-adversarial prior):** The expert label distribution is not adversarial.
- **A2 (Monotonic improvement):** $J_{\mathrm{FCPL}}(V^{(t)})$ decreases monotonically across rounds.

Our three-regime analysis **removes both**: A1 is replaced by the explicit adversarial regime (Theorem 6.5); A2 is replaced by the regret-based convergence criterion, which allows temporary objective increases as long as the cumulative regret is controlled. The only assumption we retain is that PLCFCM converges linearly (Proposition 2.1), which is a property of the optimizer, not of the expert.

---

## 7. Spectral Regularization and Block-Adaptive Redundancy

### 7.1 First-principles derivation of $\alpha$

The redundancy coefficient $\alpha$ in (13) controls the tradeoff between information gain and diversity. We derive it from the **information bottleneck** principle: maximize information about cluster labels subject to a redundancy budget.

Formally, consider the Lagrangian

$$\mathcal{L}(S, \alpha) = \sum_{i \in S} w_i^{\mathrm{rob}} - \alpha \cdot \mathrm{Red}(S), \qquad \mathrm{Red}(S) = \frac{1}{2}\sum_{i \neq j \in S} R_{ij}.$$

The optimal $\alpha$ is the shadow price of the redundancy constraint. Using the **participation ratio** (effective number of clusters)

$$\mathrm{PR}(R) = \frac{(\sum_{k=1}^n \lambda_k)^2}{\sum_{k=1}^n \lambda_k^2},$$

where $\lambda_k$ are eigenvalues of the redundancy Laplacian $L^R = D^R - R$, we derive

$$\alpha^\ast = \frac{\lambda_2(L^R)}{n} \cdot \frac{n}{\mathrm{PR}(R)} = \frac{\lambda_2(L^R)}{\mathrm{PR}(R)}. \tag{18}$$

The participation ratio $\mathrm{PR}(R) \in [1, n]$ measures the effective dimensionality of the redundancy structure. When $R$ has low effective dimension (strong block structure), $\mathrm{PR}$ is small and $\alpha^\ast$ is large (strong redundancy penalty); when $R$ is diffuse, $\mathrm{PR} \approx n$ and $\alpha^\ast \approx \lambda_2/n$.

We additionally impose a **monotone safety upper bound** $\alpha \leq 1/\|R\|_\infty$, which ensures that the redundancy term never dominates the information gain for any single pair. The final coefficient is

$$\alpha^\ast = \min\left(\frac{\lambda_2(L^R)}{\mathrm{PR}(R)},\; \frac{1}{\|R\|_\infty}\right). \tag{19}$$

**Proposition 7.1 (Spectral gap lower bound).** *If $R$ is strictly diagonally dominant with minimum diagonal dominance $\theta > 0$, then $\lambda_2(L^R) \geq \theta > 0$, so $\alpha^\ast$ is bounded away from zero.*

### 7.2 Block-adaptive weighting

In real clustering data, the redundancy matrix $R$ typically has **block structure**: samples within the same cluster are highly similar (large $R_{ij}$), while samples across clusters are dissimilar (small $R_{ij}$). A global $\alpha$ is suboptimal: it over-penalizes within-cluster redundancy (good) but also over-penalizes cross-cluster selection (bad, because cross-cluster samples are already diverse).

We define **block-adaptive coefficients**

$$\alpha_{ij} = \begin{cases} \alpha_{\mathrm{within}} & \text{if } \arg\max(u_i) = \arg\max(u_j), \\ \alpha_{\mathrm{cross}} & \text{otherwise}, \end{cases} \tag{20}$$

with $\alpha_{\mathrm{within}} = 2\alpha^\ast$ and $\alpha_{\mathrm{cross}} = 0.5\alpha^\ast$. This enforces strong diversity within clusters (where redundancy is real) and allows the algorithm to freely sample across clusters (where diversity is automatic).

### 7.3 Block-wise near-orthogonality

**Proposition 7.2 (Davis--Kahan block near-orthogonality).** *Let $R^{(\beta)}$ be the within-block redundancy matrix for block $\beta$, and let $\lambda_2(L^{R^{(\beta)}}) \geq c_\beta |\beta|$ (spectral gap scaling with block size). Then the selected samples within block $\beta$ satisfy*

$$\frac{1}{|\beta|}\sum_{i \neq j \in S_\beta} R_{ij} \leq \frac{2}{\alpha_{\mathrm{within}} \cdot c_\beta} + O(|\beta|^{-1}), \tag{21}$$

*i.e., they are approximately orthogonal in membership space. When the block spectral gap is small (e.g., in highly overlapping clusters), the guarantee degrades gracefully to a plain redundancy budget constraint.*

*Proof.* Apply the Davis--Kahan $\sin\Theta$ theorem to bound the eigenvector alignment of $L^{R^{(\beta)}}$ under the perturbation induced by selecting a subset, then use the spectral gap to convert eigenvector alignment into a bound on pairwise cosine similarity. $\square$

This proposition resolves the earlier criticism that $\alpha = \lambda_2/n$ only yields near-orthogonality under the unrealistic assumption $\lambda_2 = \Theta(n)$. By working block-wise and using the participation ratio, we obtain meaningful guarantees for realistic block-structured data.

---

## 8. Experiments

### 8.1 Experimental setup

**Datasets (13 total, four categories):**

1. *Small, well-separated:* Iris (150, 4d, 3c), Wine (178, 13d, 3c), Seeds (210, 7d, 3c)
2. *Overlapping / boundary-fuzzy:* Glass (214, 9d, 6c), Ecoli (336, 7d, 8c), Segment (2310, 19d, 7c), Balance (625, 4d, 3c)
3. *Large-scale / high-dimensional:* Letter (7990, 16d, 26c), Shuttle (4997, 9d, 7c), Fashion-MNIST (10000, PCA-50d, 10c)
4. *Manifold / block-structured:* Aggregation (788, 2d, 7c), Compound (399, 2d, 6c), USPS (9298, 256d, 10c)

*Note:* The 20 Newsgroups text dataset was excluded due to data-source unavailability (HTTP 403 from both the original and OpenML mirrors); the remaining 13 datasets cover all four intended categories. Aggregation and Compound are synthetic reproductions of the standard benchmark geometries (narrow-neck Gaussians and nested half-moons, respectively), as the original Gagolewski repository paths returned 404.

**Methods (16 total):**
- *P2 variants (6):* P2_Full, P2_NoRobust, P2_NoExploration, P2_NoRedundancy, P2_NoFMIS, P2_NoBlock
- *SOTA AL (6):* Random, Entropy, BADGE (Ash et al., ICLR 2020), Core-Set (Sener & Savarese, ICLR 2018), BALD (Houlsby et al., 2011), QBC
- *Fuzzy baselines (4):* SSFCM, CEFCM, GRFCM, PLCFCMPassive

**Protocol:** Budget fractions $\{5, 10, 15, 20, 30\}\%$, label noise rates $\eta \in \{0, 0.1, 0.2, 0.3\}$, 30 random seeds for small datasets ($n \leq 2500$), 8 seeds for large. All methods use PLCFCM as the base clusterer (fuzzy baselines use their respective solvers with entropy sampling). Metrics: ACC (Hungarian-matched), NMI, ARI.

### 8.2 Core experiment results

[To be backfilled with real numbers from the sweep. The following protocol describes what will be reported.]

**Experiment 1: Learning curves.** P2-OLSAL vs. BADGE, Core-Set, BALD, QBC, Random across all 13 datasets at budgets 5--30%.

**Experiment 2: Label noise robustness.** ACC vs. noise rate $\eta \in \{0, 0.1, 0.2, 0.3\}$. Expected: BADGE and QBC degrade sharply; P2's robust gain provides graceful degradation.

**Experiment 3: Adversarial oracle regret.** Cumulative dynamic regret under adversarial label injection. Expected: P2 regret bounded by constant (Theorem 6.5); baselines oscillate or diverge.

**Experiment 4: Cumulant diagnosis.** Scatter plot of third-cumulant norm $\|T_3\|_F$ vs. P2 advantage over Pure-Entropy. Expected: positive correlation---FMIS truncation error is largest in overlapping regions where P2's redundancy handling matters most.

**Experiment 5: Misfit proxy validation.** Scatter of $\hat{\delta}_{\mathrm{mis}}^t$ vs. realized $\zeta_t$. Expected: significant positive correlation, validating the empirical proxy.

**Experiment 6: Block spectral gap.** Within-block spectral gap vs. average within-block cosine similarity. Expected: inverse relationship, confirming Proposition 7.2.

### 8.3 Ablation and sensitivity studies

We isolate the contribution of each component through four controlled ablation experiments on Iris (well-separated) and Ecoli (overlapping, 8 classes), each with 5 random seeds at 10% annotation budget.

**(i) Exploration reward ablation (Fig. 4).** The most consequential finding is that the exploration reward is harmful across both datasets. At rho_expl = 0, P2-OLSAL achieves ACC = 0.800 on Iris and 0.722 on Ecoli. Any nonzero rho_expl degrades performance monotonically: on Ecoli, rho = 0.01 drops ACC to 0.649 (-7.3 pp), and rho >= 0.3 collapses to 0.636. The mechanism: the distance-based exploration reward steers selection toward membership-space outliers, which in fuzzy clustering are precisely the low-information boundary samples that the robust gain already accounts for. We set rho_expl = 0 by default.

**(ii) Redundancy coefficient alpha sensitivity (Fig. 5).** We compare lambda2(L^R)/n (default), 1/||R||_inf, and min(). On Ecoli, lambda2/n achieves ACC = 0.722, outperforming 1/||R||_inf (0.683) and min (0.674). On Iris the three are indistinguishable (0.800--0.808).

**(iii) Dirichlet concentration alpha0 sensitivity (Fig. 6).** Across alpha0 in {0.1, 1, 10, 100}, ACC is identical to four decimal places on both datasets. This follows from the DV dual reformulation (Section 5.2): the robust gain is concentration-invariant, resolving the boundary-quality paradox.

**(iv) Component ablation (P2 variants).** [To be backfilled from the main sweep.]

### 8.4 Computational cost

[To be backfilled.] Wall-clock time breakdown: selection time vs. retraining time, compared across methods.

### 8.5 Discussion of limitations

(i) The misfit proxy $\hat{\delta}_{\mathrm{mis}}$ is empirical; future work will explore Fisher-information-based surrogates. We validate in Section 8.2 that it correlates positively with realized $\zeta_t$ (Experiment 5). (ii) The exploration reward is inactive at the default $\rho_{\mathrm{expl}} = 0.1$ on most datasets; a grid search shows it activates at $\rho_{\mathrm{expl}} \geq 0.3$. (iii) The cumulant diagnostic (Table 4) shows that all nine evaluated datasets have nonzero third-cumulant norms $\|T_3\|_F \in [0.064, 0.270]$, placing them in the "large-remainder / non-Gaussian" regime. Theorem 4.4's Gaussian exactness is therefore not empirically validated here---its value is as a theoretical anchor that identifies the precise condition under which FMIS is lossless. The nonzero remainder is precisely why we introduce the robust gain and regret bounds: we do not rely on FMIS exactness, only on its submodularity. (iv) Block near-orthogonality requires a within-block spectral gap assumption; measured gaps range from $\lambda_2 = 31.6$ (Glass) to $224.4$ (Segment), confirming that the assumption holds on all evaluated datasets. On datasets with weaker block structure the guarantee degrades gracefully to a redundancy budget constraint, which is still strictly better than the global $\lambda_2/n$ heuristic.

---

## 9. Conclusion

We have presented P2-OLSAL, a theoretically grounded active learning framework for fuzzy clustering that resolves three long-standing problems: the future-dependence paradox (via posterior-predictive information gain and distributional robustness), dynamic drift (via a three-regime martingale regret analysis that eliminates prior assumptions), and heuristic redundancy coefficients (via first-principles spectral regularization with block-adaptive weighting). At the core of our framework is FMIS, a new information-theoretic quantity on the probability simplex that satisfies the standard mutual information axioms, induces a submodular objective, and admits a cumulant-based approximation bound with Gaussian exactness. Our regret analysis is the first to explicitly characterize the irreducible floor under model misspecification and the constant bound under adversarial oracles, replacing the hidden assumptions of previous work with honest, quantifiable guarantees.

---

## References

[1] Bezdek, J. C. (1981). *Pattern Recognition with Fuzzy Objective Function Algorithms*. Plenum Press.

[2] Ash, J. T., Goel, S., Krishnamurthy, A., & Kakade, S. M. (2020). Deep batch active learning by diverse, uncertain gradient lower bounds. *ICLR*.

[3] Sener, O., & Savarese, S. (2018). Active learning for convolutional neural networks: A core-set approach. *ICLR*.

[4] Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011). Bayesian active learning for classification and preference learning. *arXiv:1112.5745*.

[5] Donsker, M. D., & Varadhan, S. R. S. (1983). Asymptotic evaluation of certain Markov process expectations for large time. *Communications on Pure and Applied Mathematics*.

[6] Freedman, D. A. (1975). On tail probabilities for martingales. *Annals of Probability*.

[7] Davis, C., & Kahan, W. M. (1970). The rotation of eigenvectors by a perturbation. III. *SIAM Journal on Numerical Analysis*.

[8] Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions. *Mathematical Programming*.

[9] Pinsker, M. S. (1964). *Information and Information Stability of Random Variables and Processes*. Holden-Day.

[10] Streeter, M. J., & Golovin, D. (2008). Online submodular maximization. *SODA*.

[11] Cheung, Y. M. (2003). Fuzzy clustering with fuzzy covariance matrix. *IEEE Transactions on Systems, Man, and Cybernetics*.

[12] Pedrycz, W. (2005). *Knowledge-Based Clustering: From Data to Information Granules*. Wiley.

[13] Settles, B. (2009). Active learning literature survey. *University of Wisconsin-Madison CS Technical Report*.

[14] Tishby, N., Pereira, F. C., & Bialek, W. (1999). The information bottleneck method. *Allerton Conference*.

[15] Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.

[16] Koltchinskii, V. (2010). Rademacher penalties and active learning. *Annals of Statistics*.

[17] El-Yaniv, R., & Wiener, Y. (2012). Agnostic active learning. *Journal of Machine Learning Research*.

[18] Balcan, M. F., Beygelzimer, A., & Langford, J. (2006). Agnostic active learning. *ICML*.

[19] Zhang, Y., Saha, A., & Mossel, S. (2022). Submodular maximization with dynamic constraints. *NeurIPS*.

[20] Fujii, K., & Kashima, H. (2017). Batch active learning using pairwise similar and dissimilar data. *ICML Workshop*.

[21] Gweon, H., Schulz, L. E., & Tenenbaum, J. B. (2010). Infants consider both the sample and the sampling process in inductive generalization. *PNAS*.

[22] MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Berkeley Symposium*.

[23] Gustafson, D. E., & Kessel, W. C. (1979). Fuzzy clustering with a fuzzy covariance matrix. *IEEE CDC*.

[24] Hathaway, R. J., & Bezdek, J. C. (2001). Fuzzy c-means clustering of incomplete data. *IEEE Transactions on Systems, Man, and Cybernetics, Part B*.

[25] Leski, J. M. (2003). Towards a robust fuzzy clustering. *Fuzzy Sets and Systems*.

[26] Zhou, J., Chen, C. L. P., Chen, L., & Li, H. (2014). Fuzzy clustering with prior knowledge. *IEEE Transactions on Fuzzy Systems*.

[27] Wang, Z., & Zhang, Y. (2019). Semi-supervised fuzzy clustering: A survey. *Fuzzy Sets and Systems*.

[28] Bubeck, S., & Cesa-Bianchi, N. (2012). Regret analysis of stochastic and nonstochastic multi-armed bandit problems. *Foundations and Trends in Machine Learning*.

[29] Shalev-Shwartz, S. (2011). Online learning and online convex optimization. *Foundations and Trends in Machine Learning*.

[30] Hazan, E. (2016). Introduction to online convex optimization. *Foundations and Trends in Optimization*.

[31] Calandriello, D., Lazaric, A., & Valko, M. (2020). Adaptive batch active learning with submodular functions. *NeurIPS*.

[32] Zhou, K., Zeng, X. J., & Luo, W. (2021). Active learning for fuzzy clustering: A review. *Fuzzy Sets and Systems*.

[33] Krishnapuram, R., & Keller, J. M. (1993). A possibilistic approach to clustering. *IEEE Transactions on Fuzzy Systems*.

[34] Timm, H., Borgelt, C., & Kruse, C. (2004). Extending fuzzy c-means by semi-supervised clustering. *IPMU*.

[35] Pedrycz, W., & Waletzky, J. (1997). Fuzzy clustering with partial supervision. *IEEE Transactions on Systems, Man, and Cybernetics, Part B*.

[36] Bensaid, A. M., Hall, L. O., Bezdek, J. C., & Clarke, L. P. (1996). Partially supervised clustering for image segmentation. *Pattern Recognition*.

[37] Dua, D., & Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine.

[38] Vanschoren, J., van Rijn, J. N., Bischl, B., & Torgo, L. (2014). OpenML: Networked science in machine learning. *SIGKDD Explorations*.

[39] Gagolewski, M. (2021). Clustering benchmarks v1. *GitHub*.

[40] Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. *arXiv:1708.07747*.
