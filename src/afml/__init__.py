"""
AFML - Advances in Financial Machine Learning implementations.

Implements core algorithms from López de Prado (2018) since mlfinlab
is not available on Python 3.14. All implementations follow the book's
code snippets with adaptations for modern pandas/numpy.

Modules:
    fracdiff    - Fractional differentiation (Ch 5)
    labeling    - Triple-barrier method + meta-labeling (Ch 3)
    filters     - CUSUM filter for event sampling (Ch 2)
    sampling    - Sample uniqueness + sequential bootstrap (Ch 4)
    cross_val   - Purged k-fold cross-validation (Ch 7)
    importance  - Feature importance: MDI, MDA, SFI (Ch 8)
    bet_sizing  - Probability to position size (Ch 10)
    features    - Parkinson vol, Corwin-Schultz spread, entropy (Ch 18-19)
    evaluation  - RankIC, DSR, net-of-cost metrics (Ch 14)
"""
