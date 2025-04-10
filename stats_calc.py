# gui_stats_calculator_large_font_onesided.py

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import tkinter.font as tkFont # Import the font module
import math
import scipy.stats as stats

# --- Helper Functions (Mostly reused, adapted for GUI) ---

def format_output(*args):
    """Formats multiple lines for the text widget."""
    return "\n".join(map(str, args)) + "\n" + "-"*30 + "\n"

# --- Calculation Functions (Modified for Interval Type) ---

def ci_mean_z(sample_mean, sigma, n, alpha, interval_type): # Added interval_type
    try:
        if sigma <= 0 or n <= 0 or alpha <= 0 or alpha >= 1:
            return "Error: Sigma (>0), n (>0), and alpha (0<a<1) must be valid."

        if interval_type == "Two-sided":
            confidence_level = 1 - alpha
            crit_val_prob = 1 - alpha / 2
            z_critical = stats.norm.ppf(crit_val_prob)
            margin_of_error = z_critical * (sigma / math.sqrt(n))
            lower_bound = sample_mean - margin_of_error
            upper_bound = sample_mean + margin_of_error
            interval_text = f"Confidence Interval for μ: ({lower_bound:.4f}, {upper_bound:.4f})"
            formula_text = "Formula: x̄ ± Z_(α/2) * (σ/√n)"
            crit_text = f"Z-critical value (Z_alpha/2): {z_critical:.4f}"
        elif interval_type == "Lower Bound": # Interval is (L, inf)
            confidence_level = 1 - alpha
            crit_val_prob = 1 - alpha
            z_critical = stats.norm.ppf(crit_val_prob) # Use Z_alpha
            margin_of_error = z_critical * (sigma / math.sqrt(n))
            lower_bound = sample_mean - margin_of_error
            interval_text = f"Lower Confidence Bound for μ: {lower_bound:.4f} (Interval: ({lower_bound:.4f}, ∞))"
            formula_text = "Formula: x̄ - Z_α * (σ/√n)"
            crit_text = f"Z-critical value (Z_alpha): {z_critical:.4f}"
        elif interval_type == "Upper Bound": # Interval is (-inf, U)
            confidence_level = 1 - alpha
            crit_val_prob = 1 - alpha
            z_critical = stats.norm.ppf(crit_val_prob) # Use Z_alpha
            margin_of_error = z_critical * (sigma / math.sqrt(n))
            upper_bound = sample_mean + margin_of_error
            interval_text = f"Upper Confidence Bound for μ: {upper_bound:.4f} (Interval: (-∞, {upper_bound:.4f}))"
            formula_text = "Formula: x̄ + Z_α * (σ/√n)"
            crit_text = f"Z-critical value (Z_alpha): {z_critical:.4f}"
        else:
            return "Error: Invalid interval type."

        return format_output(
            f"--- CI for Mean (Z-distribution) - {interval_type} ---",
            f"Confidence Level: {confidence_level*100:.2f}% (α={alpha})",
            f"Inputs: x̄={sample_mean}, σ={sigma}, n={n}",
            crit_text,
            f"Margin of Error (used): {margin_of_error:.4f}",
            interval_text,
            formula_text
        )
    except Exception as e:
        return f"Error: {e}"

def ci_mean_t(sample_mean, sample_std, n, alpha, interval_type): # Added interval_type
    try:
        if sample_std <= 0 or n <= 1 or alpha <= 0 or alpha >= 1:
             return "Error: Sample std dev (>0), n (>1), and alpha (0<a<1) must be valid."
        df = n - 1

        if interval_type == "Two-sided":
            confidence_level = 1 - alpha
            crit_val_prob = 1 - alpha / 2
            t_critical = stats.t.ppf(crit_val_prob, df)
            margin_of_error = t_critical * (sample_std / math.sqrt(n))
            lower_bound = sample_mean - margin_of_error
            upper_bound = sample_mean + margin_of_error
            interval_text = f"Confidence Interval for μ: ({lower_bound:.4f}, {upper_bound:.4f})"
            formula_text = "Formula: x̄ ± t_(α/2, n-1) * (s/√n)"
            crit_text = f"t-critical value (t_alpha/2, n-1): {t_critical:.4f}"
        elif interval_type == "Lower Bound": # Interval is (L, inf)
            confidence_level = 1 - alpha
            crit_val_prob = 1 - alpha
            t_critical = stats.t.ppf(crit_val_prob, df) # Use t_alpha
            margin_of_error = t_critical * (sample_std / math.sqrt(n))
            lower_bound = sample_mean - margin_of_error
            interval_text = f"Lower Confidence Bound for μ: {lower_bound:.4f} (Interval: ({lower_bound:.4f}, ∞))"
            formula_text = "Formula: x̄ - t_(α, n-1) * (s/√n)"
            crit_text = f"t-critical value (t_alpha, n-1): {t_critical:.4f}"
        elif interval_type == "Upper Bound": # Interval is (-inf, U)
            confidence_level = 1 - alpha
            crit_val_prob = 1 - alpha
            t_critical = stats.t.ppf(crit_val_prob, df) # Use t_alpha
            margin_of_error = t_critical * (sample_std / math.sqrt(n))
            upper_bound = sample_mean + margin_of_error
            interval_text = f"Upper Confidence Bound for μ: {upper_bound:.4f} (Interval: (-∞, {upper_bound:.4f}))"
            formula_text = "Formula: x̄ + t_(α, n-1) * (s/√n)"
            crit_text = f"t-critical value (t_alpha, n-1): {t_critical:.4f}"
        else:
            return "Error: Invalid interval type."

        return format_output(
            f"--- CI for Mean (t-distribution) - {interval_type} ---",
            f"Confidence Level: {confidence_level*100:.2f}% (α={alpha})",
            f"Inputs: x̄={sample_mean}, s={sample_std}, n={n}",
            f"Degrees of Freedom: {df}",
            crit_text,
            f"Margin of Error (used): {margin_of_error:.4f}",
            interval_text,
            formula_text
        )
    except Exception as e:
        return f"Error: {e}"

def ci_variance(sample_variance, n, alpha, interval_type): # Added interval_type
     try:
        if sample_variance < 0 or n <= 1 or alpha <= 0 or alpha >= 1:
             return "Error: Sample variance (>=0), n (>1), and alpha (0<a<1) must be valid."
        df = n - 1
        confidence_level = 1 - alpha

        if interval_type == "Two-sided":
            chi2_lower_crit_for_upper_bound = stats.chi2.ppf(alpha / 2, df) # Lower tail area for Upper Bound
            chi2_upper_crit_for_lower_bound = stats.chi2.ppf(1 - alpha / 2, df) # Upper tail area for Lower Bound

            if chi2_upper_crit_for_lower_bound <= 1e-9: return "Error: Upper Chi-Square critical value is too close to zero."

            lower_bound = (n - 1) * sample_variance / chi2_upper_crit_for_lower_bound
            upper_bound = (n - 1) * sample_variance / chi2_lower_crit_for_upper_bound

            interval_text = f"Confidence Interval for σ²: ({lower_bound:.4f}, {upper_bound:.4f})"
            interval_text_s = f"CI for σ: ({math.sqrt(lower_bound):.4f}, {math.sqrt(upper_bound):.4f})"
            crit_text = (f"Chi2 critical (for UB, χ²_(1-α/2,{df})): {chi2_lower_crit_for_upper_bound:.4f}\n"
                         f"Chi2 critical (for LB, χ²_(α/2,{df})): {chi2_upper_crit_for_lower_bound:.4f}")
            formula_text = "Formula: [(n-1)s²/χ²_(α/2,n-1), (n-1)s²/χ²_(1-α/2,n-1)]"

        elif interval_type == "Lower Bound": # Interval (L, inf) for variance
            chi2_upper_crit_for_lower_bound = stats.chi2.ppf(1 - alpha, df) # χ²_α
            if chi2_upper_crit_for_lower_bound <= 1e-9: return "Error: Upper Chi-Square critical value is too close to zero."
            lower_bound = (n - 1) * sample_variance / chi2_upper_crit_for_lower_bound
            interval_text = f"Lower Confidence Bound for σ²: {lower_bound:.4f} (Interval: ({lower_bound:.4f}, ∞))"
            interval_text_s = f"Lower Confidence Bound for σ: {math.sqrt(lower_bound):.4f}"
            crit_text = f"Chi2 critical (χ²_(α, {df})): {chi2_upper_crit_for_lower_bound:.4f}"
            formula_text = "Formula LB = (n-1)s² / χ²_(α, n-1)"

        elif interval_type == "Upper Bound": # Interval (0, U) for variance
             chi2_lower_crit_for_upper_bound = stats.chi2.ppf(alpha, df) # χ²_(1-α)
             if chi2_lower_crit_for_upper_bound <= 1e-9: # Avoid division by zero
                 upper_bound = float('inf')
             else:
                 upper_bound = (n - 1) * sample_variance / chi2_lower_crit_for_upper_bound

             interval_text = f"Upper Confidence Bound for σ²: {upper_bound:.4f} (Interval: (0, {upper_bound:.4f}))"
             interval_text_s = f"Upper Confidence Bound for σ: {math.sqrt(upper_bound):.4f}"
             crit_text = f"Chi2 critical (χ²_(1-α, {df})): {chi2_lower_crit_for_upper_bound:.4f}"
             formula_text = "Formula UB = (n-1)s² / χ²_(1-α, n-1)"
        else:
             return "Error: Invalid interval type."

        return format_output(
            f"--- CI for Variance (Chi-Square) - {interval_type} ---",
            f"Confidence Level: {confidence_level*100:.2f}% (α={alpha})",
            f"Inputs: s²={sample_variance}, n={n}",
            f"Degrees of Freedom: {df}",
            crit_text,
            interval_text,
            interval_text_s,
            formula_text
        )
     except Exception as e:
         return f"Error: {e}"

def ci_proportion(x, n, alpha, interval_type): # Added interval_type
    try:
        if x < 0 or n <= 0 or x > n or alpha <= 0 or alpha >= 1:
            return "Error: x (>=0), n (>0), x <= n, and alpha (0<a<1) must be valid."
        phat = x / n
        warning = ""
        if not (n * phat >= 5 and n * (1 - phat) >= 5):
            warning = "Warning: Sample size may be small for normal approximation (np̂ < 5 or n(1-p̂) < 5).\n"

        # Use Wald interval for consistency with notes, despite limitations
        try:
            se = math.sqrt(phat * (1 - phat) / n)
        except ValueError: # phat=0 or phat=1
            se = 0 # Margin of error will be 0 for one-sided in this case
            warning += "Note: p̂ is 0 or 1, Wald interval bounds may be trivial.\n"

        if interval_type == "Two-sided":
            confidence_level = 1 - alpha
            z_critical = stats.norm.ppf(1 - alpha / 2)
            margin_of_error = z_critical * se
            lower_bound = phat - margin_of_error
            upper_bound = phat + margin_of_error
            lower_bound = max(0, lower_bound)
            upper_bound = min(1, upper_bound)
            interval_text = f"Confidence Interval for p: ({lower_bound:.4f}, {upper_bound:.4f})"
            formula_text = "Formula: p̂ ± Z_(α/2) * √[p̂(1-p̂)/n]"
            crit_text = f"Z-critical value (Z_alpha/2): {z_critical:.4f}"
            moe_text = f"Margin of Error: {margin_of_error:.4f}"
        elif interval_type == "Lower Bound": # Interval (L, 1)
            confidence_level = 1 - alpha
            z_critical = stats.norm.ppf(1 - alpha) # Z_alpha
            margin_of_error = z_critical * se
            lower_bound = phat - margin_of_error
            lower_bound = max(0, lower_bound)
            interval_text = f"Lower Confidence Bound for p: {lower_bound:.4f} (Interval: ({lower_bound:.4f}, 1])"
            formula_text = "Formula: p̂ - Z_α * √[p̂(1-p̂)/n]"
            crit_text = f"Z-critical value (Z_alpha): {z_critical:.4f}"
            moe_text = f"Margin of Error (used): {margin_of_error:.4f}"
        elif interval_type == "Upper Bound": # Interval (0, U)
            confidence_level = 1 - alpha
            z_critical = stats.norm.ppf(1 - alpha) # Z_alpha
            margin_of_error = z_critical * se
            upper_bound = phat + margin_of_error
            upper_bound = min(1, upper_bound)
            interval_text = f"Upper Confidence Bound for p: {upper_bound:.4f} (Interval: [0, {upper_bound:.4f}))"
            formula_text = "Formula: p̂ + Z_α * √[p̂(1-p̂)/n]"
            crit_text = f"Z-critical value (Z_alpha): {z_critical:.4f}"
            moe_text = f"Margin of Error (used): {margin_of_error:.4f}"
        else:
            return "Error: Invalid interval type."

        return format_output(
            f"--- CI for Proportion (Z-distribution, Wald) - {interval_type} ---",
            warning,
            f"Confidence Level: {confidence_level*100:.2f}% (α={alpha})",
            f"Inputs: x={x}, n={n}",
            f"Sample Proportion (p̂): {phat:.4f}",
            crit_text,
            moe_text,
            interval_text,
            formula_text
        )
    except Exception as e:
        return f"Error: {e}"

# --- Hypothesis Test and Sample Size functions remain unchanged ---
def ht_mean_z(sample_mean, mu0, sigma, n, alpha, alternative):
    try:
        if sigma <= 0 or n <= 0 or alpha <= 0 or alpha >= 1:
            return "Error: Sigma (>0), n (>0), and alpha (0<a<1) must be valid."
        z_stat = (sample_mean - mu0) / (sigma / math.sqrt(n))
        reject = False
        critical_val_text = ""

        if alternative == 'two-sided':
            p_value = 2 * stats.norm.sf(abs(z_stat))
            critical_val = stats.norm.ppf(1 - alpha / 2)
            reject = abs(z_stat) >= critical_val
            critical_val_text = f"Critical Values: ±{critical_val:.4f}"
            reject_rule = f"Reject H₀ if |Z₀| ≥ {critical_val:.4f}"
        elif alternative == 'less':
            p_value = stats.norm.cdf(z_stat)
            critical_val = stats.norm.ppf(alpha)
            reject = z_stat <= critical_val
            critical_val_text = f"Critical Value: {critical_val:.4f}"
            reject_rule = f"Reject H₀ if Z₀ ≤ {critical_val:.4f}"
        else: # greater
            p_value = stats.norm.sf(z_stat)
            critical_val = stats.norm.ppf(1 - alpha)
            reject = z_stat >= critical_val
            critical_val_text = f"Critical Value: {critical_val:.4f}"
            reject_rule = f"Reject H₀ if Z₀ ≥ {critical_val:.4f}"

        return format_output(
            f"--- Hypothesis Test for Mean (Z-distribution) ---",
            f"H₀: μ = {mu0}",
            f"H₁: μ { {'two-sided':'!=', 'less':'<', 'greater':'>'}[alternative] } {mu0}",
            f"Inputs: x̄={sample_mean}, μ₀={mu0}, σ={sigma}, n={n}, α={alpha}",
            f"Test Statistic (Z₀): {z_stat:.4f}",
            f"P-value: {p_value:.4f}",
            critical_val_text,
            reject_rule,
            f"Decision: {'Reject H₀' if reject else 'Fail to reject H₀'}",
            f"Formula (Test Stat): Z₀ = (x̄ - μ₀) / (σ/√n)"
        )
    except Exception as e:
        return f"Error: {e}"

def ht_mean_t(sample_mean, mu0, sample_std, n, alpha, alternative):
    try:
        if sample_std <= 0 or n <= 1 or alpha <= 0 or alpha >= 1:
             return "Error: Sample std dev (>0), n (>1), and alpha (0<a<1) must be valid."
        df = n - 1
        t_stat = (sample_mean - mu0) / (sample_std / math.sqrt(n))
        reject = False
        critical_val_text = ""

        if alternative == 'two-sided':
            p_value = 2 * stats.t.sf(abs(t_stat), df)
            critical_val = stats.t.ppf(1 - alpha / 2, df)
            reject = abs(t_stat) >= critical_val
            critical_val_text = f"Critical Values: ±{critical_val:.4f}"
            reject_rule = f"Reject H₀ if |t₀| ≥ {critical_val:.4f}"
        elif alternative == 'less':
            p_value = stats.t.cdf(t_stat, df)
            critical_val = stats.t.ppf(alpha, df)
            reject = t_stat <= critical_val
            critical_val_text = f"Critical Value: {critical_val:.4f}"
            reject_rule = f"Reject H₀ if t₀ ≤ {critical_val:.4f}"
        else: # greater
            p_value = stats.t.sf(t_stat, df)
            critical_val = stats.t.ppf(1 - alpha, df)
            reject = t_stat >= critical_val
            critical_val_text = f"Critical Value: {critical_val:.4f}"
            reject_rule = f"Reject H₀ if t₀ ≥ {critical_val:.4f}"

        return format_output(
            f"--- Hypothesis Test for Mean (t-distribution) ---",
            f"H₀: μ = {mu0}",
            f"H₁: μ { {'two-sided':'!=', 'less':'<', 'greater':'>'}[alternative] } {mu0}",
            f"Inputs: x̄={sample_mean}, μ₀={mu0}, s={sample_std}, n={n}, α={alpha}",
            f"Degrees of Freedom: {df}",
            f"Test Statistic (t₀): {t_stat:.4f}",
            f"P-value: {p_value:.4f}",
            critical_val_text,
            reject_rule,
            f"Decision: {'Reject H₀' if reject else 'Fail to reject H₀'}",
            f"Formula (Test Stat): t₀ = (x̄ - μ₀) / (s/√n)"
        )
    except Exception as e:
        return f"Error: {e}"

def ht_variance(sample_variance, sigma0_sq, n, alpha, alternative):
     try:
        if sample_variance < 0 or sigma0_sq <= 0 or n <= 1 or alpha <= 0 or alpha >= 1:
            return "Error: Sample variance (>=0), Hypothesized variance (>0), n (>1), and alpha (0<a<1) must be valid."
        df = n - 1
        chi2_stat = (n - 1) * sample_variance / sigma0_sq
        reject = False
        critical_val_text = ""

        if alternative == 'two-sided':
            p_lower = stats.chi2.cdf(chi2_stat, df)
            p_upper = stats.chi2.sf(chi2_stat, df)
            p_value = 2 * min(p_lower, p_upper)
            crit_lower = stats.chi2.ppf(alpha / 2, df)
            crit_upper = stats.chi2.ppf(1 - alpha / 2, df)
            reject = chi2_stat <= crit_lower or chi2_stat >= crit_upper
            critical_val_text = f"Critical Values: {crit_lower:.4f} (lower), {crit_upper:.4f} (upper)"
            reject_rule = f"Reject H₀ if χ²₀ ≤ {crit_lower:.4f} or χ²₀ ≥ {crit_upper:.4f}"
        elif alternative == 'less':
            p_value = stats.chi2.cdf(chi2_stat, df)
            critical_val = stats.chi2.ppf(alpha, df) # Lower tail critical value
            reject = chi2_stat <= critical_val
            critical_val_text = f"Critical Value: {critical_val:.4f}"
            reject_rule = f"Reject H₀ if χ²₀ ≤ {critical_val:.4f} (χ²_(1-α,{df}))"
        else: # greater
            p_value = stats.chi2.sf(chi2_stat, df)
            critical_val = stats.chi2.ppf(1 - alpha, df) # Upper tail critical value
            reject = chi2_stat >= critical_val
            critical_val_text = f"Critical Value: {critical_val:.4f}"
            reject_rule = f"Reject H₀ if χ²₀ ≥ {critical_val:.4f} (χ²_(α,{df}))"

        return format_output(
            f"--- Hypothesis Test for Variance (Chi-Square) ---",
            f"H₀: σ² = {sigma0_sq}",
            f"H₁: σ² { {'two-sided':'!=', 'less':'<', 'greater':'>'}[alternative] } {sigma0_sq}",
            f"Inputs: s²={sample_variance}, σ₀²={sigma0_sq}, n={n}, α={alpha}",
            f"Degrees of Freedom: {df}",
            f"Test Statistic (χ²₀): {chi2_stat:.4f}",
            f"P-value: {p_value:.4f}",
            critical_val_text,
            reject_rule,
            f"Decision: {'Reject H₀' if reject else 'Fail to reject H₀'}",
            f"Formula (Test Stat): χ²₀ = (n-1)s² / σ₀²"
        )
     except Exception as e:
         return f"Error: {e}"

def ht_proportion(x, n, p0, alpha, alternative):
    try:
        if x < 0 or n <= 0 or x > n or p0 <= 0 or p0 >= 1 or alpha <= 0 or alpha >= 1:
             return "Error: x (>=0), n (>0), x <= n, p0 (0<p0<1), and alpha (0<a<1) must be valid."
        warning = ""
        if not (n * p0 >= 5 and n * (1 - p0) >= 5):
             warning = "Warning: Sample size may be small for normal approximation based on H₀ (np₀ < 5 or n(1-p₀) < 5).\n"

        phat = x / n
        try:
            se = math.sqrt(p0 * (1 - p0) / n)
            if se == 0: return "Error: Standard error is zero based on p0."
            z_stat = (phat - p0) / se
        except ZeroDivisionError:
             return "Error calculating standard error. Check inputs (n>0)."
        except ValueError: # math domain error if p0*(1-p0) somehow negative (shouldn't happen)
             return "Error calculating standard error (invalid p0)."


        reject = False
        critical_val_text = ""
        if alternative == 'two-sided':
            p_value = 2 * stats.norm.sf(abs(z_stat))
            critical_val = stats.norm.ppf(1 - alpha / 2)
            reject = abs(z_stat) >= critical_val
            critical_val_text = f"Critical Values: ±{critical_val:.4f}"
            reject_rule = f"Reject H₀ if |Z₀| ≥ {critical_val:.4f}"
        elif alternative == 'less':
            p_value = stats.norm.cdf(z_stat)
            critical_val = stats.norm.ppf(alpha)
            reject = z_stat <= critical_val
            critical_val_text = f"Critical Value: {critical_val:.4f}"
            reject_rule = f"Reject H₀ if Z₀ ≤ {critical_val:.4f}"
        else: # greater
            p_value = stats.norm.sf(z_stat)
            critical_val = stats.norm.ppf(1 - alpha)
            reject = z_stat >= critical_val
            critical_val_text = f"Critical Value: {critical_val:.4f}"
            reject_rule = f"Reject H₀ if Z₀ ≥ {critical_val:.4f}"

        return format_output(
            f"--- Hypothesis Test for Proportion (Z-distribution) ---",
            warning,
            f"H₀: p = {p0}",
            f"H₁: p { {'two-sided':'!=', 'less':'<', 'greater':'>'}[alternative] } {p0}",
            f"Inputs: x={x}, n={n}, p₀={p0}, α={alpha}",
            f"Sample Proportion (p̂): {phat:.4f}",
            f"Test Statistic (Z₀): {z_stat:.4f}",
            f"P-value: {p_value:.4f}",
            critical_val_text,
            reject_rule,
            f"Decision: {'Reject H₀' if reject else 'Fail to reject H₀'}",
            f"Formula (Test Stat): Z₀ = (p̂ - p₀) / √[p₀(1-p₀)/n]"
        )
    except Exception as e:
        return f"Error: {e}"

def sample_size_mean(sigma, error, alpha):
     try:
        if sigma <= 0 or error <= 0 or alpha <= 0 or alpha >= 1:
            return "Error: Sigma (>0), error margin (>0), and alpha (0<a<1) must be valid."
        z_critical = stats.norm.ppf(1 - alpha / 2)
        n = ((z_critical * sigma) / error) ** 2
        n_required = math.ceil(n)
        return format_output(
            f"--- Sample Size Calculation (Mean CI) ---",
            f"Required Margin of Error (E): {error}",
            f"Desired Confidence: {(1-alpha)*100:.1f}% (α={alpha})",
            f"Assumed Population Std Dev (σ): {sigma}",
            f"Z-critical value (Z_alpha/2): {z_critical:.4f}",
            f"Calculated n (raw): {n:.4f}",
            f"Required Sample Size (n): {n_required}",
            f"Formula: n = (Z_(α/2) * σ / E)²"
        )
     except Exception as e:
         return f"Error: {e}"

def sample_size_proportion(p_hat_estimate, error, alpha):
    try:
        if error <= 0 or alpha <= 0 or alpha >= 1:
             return "Error: Error margin (>0), and alpha (0<a<1) must be valid."
        if p_hat_estimate is None:
            p_hat = 0.5 # Conservative estimate
            est_text = "Using conservative estimate p̂ = 0.5"
        elif p_hat_estimate < 0 or p_hat_estimate > 1:
            return "Error: Proportion estimate must be between 0 and 1."
        else:
            p_hat = p_hat_estimate
            est_text = f"Using estimated proportion p̂ = {p_hat:.3f}"

        z_critical = stats.norm.ppf(1 - alpha / 2)
        n = ((z_critical / error) ** 2) * p_hat * (1 - p_hat)
        n_required = math.ceil(n)

        return format_output(
            f"--- Sample Size Calculation (Proportion CI) ---",
            est_text,
            f"Required Margin of Error (E): {error}",
            f"Desired Confidence: {(1-alpha)*100:.1f}% (α={alpha})",
            f"Z-critical value (Z_alpha/2): {z_critical:.4f}",
            f"Calculated n (raw): {n:.4f}",
            f"Required Sample Size (n): {n_required}",
            f"Formula: n = (Z_(α/2) / E)² * p̂(1-p̂)"
        )
    except Exception as e:
        return f"Error: {e}"

def calculate_critical_value(dist_type, tail_type, alpha_val, df=None):
    try:
        if alpha_val <= 0 or alpha_val >= 1:
            return "Error: Alpha must be between 0 and 1."

        crit_val = None
        description = ""

        if dist_type == "Z":
            if tail_type == "Two-tailed (±Zα/2)":
                crit_val = stats.norm.ppf(1 - alpha_val / 2)
                description = f"Z_(α/2) for α={alpha_val}"
                result = f"±{crit_val:.4f}"
            elif tail_type == "Upper tail (Zα)":
                crit_val = stats.norm.ppf(1 - alpha_val)
                description = f"Z_α for upper tail area α={alpha_val}"
                result = f"{crit_val:.4f}"
            elif tail_type == "Lower tail (-Zα)":
                crit_val = stats.norm.ppf(alpha_val) # This is already negative
                description = f"-Z_α for lower tail area α={alpha_val}"
                result = f"{crit_val:.4f}"
            else: return "Invalid tail type for Z."

        elif dist_type == "t":
            if df is None or df <= 0: return "Error: Valid df (>0) required for t-distribution."
            if tail_type == "Two-tailed (±tα/2)":
                crit_val = stats.t.ppf(1 - alpha_val / 2, df)
                description = f"t_(α/2, {df}) for α={alpha_val}"
                result = f"±{crit_val:.4f}"
            elif tail_type == "Upper tail (tα)":
                crit_val = stats.t.ppf(1 - alpha_val, df)
                description = f"t_(α, {df}) for upper tail area α={alpha_val}"
                result = f"{crit_val:.4f}"
            elif tail_type == "Lower tail (-tα)":
                crit_val = stats.t.ppf(alpha_val, df) # This is already negative
                description = f"-t_(α, {df}) for lower tail area α={alpha_val}"
                result = f"{crit_val:.4f}"
            else: return "Invalid tail type for t."

        elif dist_type == "Chi-square (χ²)":
             if df is None or df <= 0: return "Error: Valid df (>0) required for Chi-square."
             if tail_type == "Upper tail (χ²α)": # e.g., χ²_(0.05)
                 crit_val = stats.chi2.ppf(1 - alpha_val, df)
                 description = f"χ²_(α, {df}) for upper tail area α={alpha_val}"
                 result = f"{crit_val:.4f}"
             elif tail_type == "Lower tail (χ²(1-α))": # e.g., χ²_(0.95)
                 crit_val = stats.chi2.ppf(alpha_val, df)
                 description = f"χ²_(1-α, {df}) for lower tail area α={alpha_val}"
                 result = f"{crit_val:.4f}"
             elif tail_type == "Upper tail CI (χ²α/2)": # Upper critical for CI (lower bound of interval)
                  crit_val = stats.chi2.ppf(1 - alpha_val / 2, df)
                  description = f"χ²_(α/2, {df}) for CI, α={alpha_val}"
                  result = f"{crit_val:.4f}"
             elif tail_type == "Lower tail CI (χ²(1-α/2))": # Lower critical for CI (upper bound of interval)
                  crit_val = stats.chi2.ppf(alpha_val / 2, df)
                  description = f"χ²_(1-α/2, {df}) for CI, α={alpha_val}"
                  result = f"{crit_val:.4f}"
             else: return "Invalid tail type for Chi-square."
        else:
            return "Error: Unknown distribution type."

        return format_output(
            f"--- Critical Value Calculation ---",
            f"Distribution: {dist_type}",
            f"Type: {description}",
            f"Degrees of Freedom (df): {df if df else 'N/A'}",
            f"Calculated Critical Value(s): {result}"
        )

    except Exception as e:
        return f"Error: {e}"


# --- GUI Class ---
class StatsCalculatorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Statistics Calculator")
        master.geometry("750x700") # Keep slightly larger size

        # --- Font Definitions ---
        self.default_font = tkFont.Font(family="Helvetica", size=12)
        self.label_font = tkFont.Font(family="Helvetica", size=11)
        self.entry_font = tkFont.Font(family="Helvetica", size=11)
        self.button_font = tkFont.Font(family="Helvetica", size=12, weight="bold")
        self.results_font = tkFont.Font(family="Courier New", size=11)
        self.tab_font = tkFont.Font(family="Helvetica", size=11, weight="bold")
        self.header_font = tkFont.Font(family="Helvetica", size=11, weight="bold")

        # --- Style Configuration ---
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.style.configure('.', font=self.default_font)
        self.style.configure('TLabel', font=self.label_font)
        self.style.configure('TButton', font=self.button_font, padding=5)
        self.style.configure('TEntry', font=self.entry_font)
        self.style.configure('TCombobox', font=self.entry_font)
        self.style.configure('TRadiobutton', font=self.label_font)
        self.style.configure('TNotebook.Tab', font=self.tab_font, padding=[10, 5])
        self.style.configure('TLabelframe.Label', font=self.header_font)

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(master)

        # Create Tabs
        self.tab_ci = ttk.Frame(self.notebook, padding="10")
        self.tab_ht = ttk.Frame(self.notebook, padding="10")
        self.tab_ss = ttk.Frame(self.notebook, padding="10")
        self.tab_cv = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(self.tab_ci, text='Confidence Intervals')
        self.notebook.add(self.tab_ht, text='Hypothesis Tests')
        self.notebook.add(self.tab_ss, text='Sample Size')
        self.notebook.add(self.tab_cv, text='Critical Values')

        self.notebook.pack(expand=True, fill='both', padx=10, pady=5)

        # Populate Tabs
        self.create_ci_tab()
        self.create_ht_tab()
        self.create_ss_tab()
        self.create_cv_tab()

        # Results Area
        self.results_text = tk.Text(master, height=14, width=85, wrap=tk.WORD,
                                   relief=tk.SUNKEN, borderwidth=1, font=self.results_font)
        self.results_text.pack(pady=(5, 10), padx=10, fill='x', expand=False)
        self.results_text.config(state=tk.DISABLED)

    def display_results(self, results):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert(tk.END, results)
        self.results_text.config(state=tk.DISABLED)

    def get_float(self, entry_widget, default=None):
        try:
            return float(entry_widget.get())
        except ValueError:
            return default

    def get_int(self, entry_widget, default=None):
         try:
             val = entry_widget.get()
             if not val: return default
             return int(val)
         except ValueError:
             return default

    # --- Tab Creation Methods ---

    def create_ci_tab(self):
        frame = self.tab_ci
        widgets = {}

        # --- Row 0: Parameter and Interval Type ---
        ttk.Label(frame, text="Parameter:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        widgets['param'] = ttk.Combobox(frame, values=["Mean (μ)", "Variance (σ²)", "Proportion (p)"], state="readonly", width=18)
        widgets['param'].grid(row=0, column=1, padx=5, pady=5, sticky='w') # Use sticky w, not ew
        widgets['param'].set("Mean (μ)")
        widgets['param'].bind("<<ComboboxSelected>>", lambda e: self.update_ci_inputs(widgets))

        ttk.Label(frame, text="Interval Type:").grid(row=0, column=2, padx=5, pady=5, sticky='w')
        widgets['interval_type'] = ttk.Combobox(frame, values=["Two-sided", "Lower Bound", "Upper Bound"], state="readonly", width=15)
        widgets['interval_type'].grid(row=0, column=3, padx=5, pady=5, sticky='w') # Use sticky w
        widgets['interval_type'].set("Two-sided")

        # --- Row 1: Input Frame ---
        input_frame = ttk.LabelFrame(frame, text="Inputs")
        input_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky='ew')
        widgets['input_frame'] = input_frame

        # Dynamic Input Area
        self.update_ci_inputs(widgets, initial=True) # Pass widgets dict

        # --- Row 2: Calculate Button ---
        calc_button = ttk.Button(frame, text="Calculate CI", command=lambda: self.calculate_ci(widgets)) # Pass widgets dict
        calc_button.grid(row=2, column=0, columnspan=4, pady=15)

    def update_ci_inputs(self, widgets, initial=False): # Now receives widgets dict
        param = widgets['param'].get()
        frame = widgets['input_frame']
        # Clear previous widgets
        for widget in frame.winfo_children():
            widget.destroy()

        # Common Alpha Input (Row 0 within input_frame)
        ttk.Label(frame, text="Alpha (α):").grid(row=0, column=0, padx=5, pady=3, sticky='w')
        widgets['alpha'] = ttk.Entry(frame, width=10)
        widgets['alpha'].grid(row=0, column=1, padx=5, pady=3, sticky='w')
        widgets['alpha'].insert(0, "0.05")


        if param == "Mean (μ)":
            # Mean specific inputs (Start from row 1 within input_frame)
            ttk.Label(frame, text="Method:").grid(row=1, column=0, padx=5, pady=3, sticky='w')
            widgets['mean_method_var'] = tk.StringVar(value="Z")
            ttk.Radiobutton(frame, text="Z (σ known or n>30)", variable=widgets['mean_method_var'], value="Z", command=lambda: self.toggle_sigma_s(widgets)).grid(row=2, column=0, columnspan=2, padx=5, sticky='w')
            ttk.Radiobutton(frame, text="t (σ unknown, n≤30)", variable=widgets['mean_method_var'], value="t", command=lambda: self.toggle_sigma_s(widgets)).grid(row=3, column=0, columnspan=2, padx=5, sticky='w')

            ttk.Label(frame, text="Sample Mean (x̄):").grid(row=4, column=0, padx=5, pady=3, sticky='w')
            widgets['mean_xbar'] = ttk.Entry(frame, width=10)
            widgets['mean_xbar'].grid(row=4, column=1, padx=5, pady=3, sticky='w')

            ttk.Label(frame, text="Sample Size (n):").grid(row=5, column=0, padx=5, pady=3, sticky='w')
            widgets['mean_n'] = ttk.Entry(frame, width=10)
            widgets['mean_n'].grid(row=5, column=1, padx=5, pady=3, sticky='w')

            # Sigma / S input area (Start from row 6)
            widgets['sigma_label'] = ttk.Label(frame, text="Pop. Std Dev (σ):")
            widgets['sigma_entry'] = ttk.Entry(frame, width=10)
            widgets['s_label'] = ttk.Label(frame, text="Sample Std Dev (s):")
            widgets['s_entry'] = ttk.Entry(frame, width=10)

            # Place sigma/s in the next available row initially
            self.toggle_sigma_s(widgets, start_row=6)

        elif param == "Variance (σ²)":
            ttk.Label(frame, text="Sample Variance (s²):").grid(row=1, column=0, padx=5, pady=3, sticky='w')
            widgets['var_s2'] = ttk.Entry(frame, width=10)
            widgets['var_s2'].grid(row=1, column=1, padx=5, pady=3, sticky='w')
            ttk.Label(frame, text="Sample Size (n):").grid(row=2, column=0, padx=5, pady=3, sticky='w')
            widgets['var_n'] = ttk.Entry(frame, width=10)
            widgets['var_n'].grid(row=2, column=1, padx=5, pady=3, sticky='w')

        elif param == "Proportion (p)":
            ttk.Label(frame, text="Successes (x):").grid(row=1, column=0, padx=5, pady=3, sticky='w')
            widgets['prop_x'] = ttk.Entry(frame, width=10)
            widgets['prop_x'].grid(row=1, column=1, padx=5, pady=3, sticky='w')
            ttk.Label(frame, text="Sample Size (n):").grid(row=2, column=0, padx=5, pady=3, sticky='w')
            widgets['prop_n'] = ttk.Entry(frame, width=10)
            widgets['prop_n'].grid(row=2, column=1, padx=5, pady=3, sticky='w')


    def toggle_sigma_s(self, widgets, start_row=6): # Pass start_row
         frame = widgets['input_frame']
         method = widgets['mean_method_var'].get()
         # Use pady=3 and the start_row
         if method == "Z":
             widgets['s_label'].grid_remove()
             widgets['s_entry'].grid_remove()
             widgets['sigma_label'].grid(row=start_row, column=0, padx=5, pady=3, sticky='w')
             widgets['sigma_entry'].grid(row=start_row, column=1, padx=5, pady=3, sticky='w')
         else: # method == "t"
             widgets['sigma_label'].grid_remove()
             widgets['sigma_entry'].grid_remove()
             widgets['s_label'].grid(row=start_row, column=0, padx=5, pady=3, sticky='w')
             widgets['s_entry'].grid(row=start_row, column=1, padx=5, pady=3, sticky='w')


    def calculate_ci(self, widgets): # Receives widgets dict
        try:
            param = widgets['param'].get()
            interval_type = widgets['interval_type'].get() # Get interval type
            alpha = self.get_float(widgets['alpha'], -1)
            result = "Error: Invalid inputs." # Default error

            if param == "Mean (μ)":
                method = widgets['mean_method_var'].get()
                mean = self.get_float(widgets['mean_xbar'])
                n = self.get_int(widgets['mean_n'])
                if n is None or mean is None: raise ValueError("Mean and N required.")

                if method == "Z":
                    sigma = self.get_float(widgets['sigma_entry'])
                    if sigma is None: raise ValueError("Sigma required for Z method.")
                    result = ci_mean_z(mean, sigma, n, alpha, interval_type) # Pass interval_type
                else: # method == "t"
                    s = self.get_float(widgets['s_entry'])
                    if s is None: raise ValueError("Sample Std Dev (s) required for t method.")
                    result = ci_mean_t(mean, s, n, alpha, interval_type) # Pass interval_type

            elif param == "Variance (σ²)":
                 s2 = self.get_float(widgets['var_s2'])
                 n = self.get_int(widgets['var_n'])
                 if s2 is None or n is None: raise ValueError("Variance and N required.")
                 result = ci_variance(s2, n, alpha, interval_type) # Pass interval_type

            elif param == "Proportion (p)":
                 x = self.get_int(widgets['prop_x'])
                 n = self.get_int(widgets['prop_n'])
                 if x is None or n is None: raise ValueError("X and N required.")
                 result = ci_proportion(x, n, alpha, interval_type) # Pass interval_type

            self.display_results(result)

        except Exception as e:
             messagebox.showerror("Input Error", f"Could not perform calculation.\nCheck inputs.\nDetails: {e}")
             self.display_results(f"Error during calculation: {e}")

    # --- HT, SS, CV Tabs and Methods remain unchanged ---
    def create_ht_tab(self):
        # This function remains the same as the previous version
        frame = self.tab_ht
        widgets = {}

        # Parameter Selection
        ttk.Label(frame, text="Parameter:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        widgets['param'] = ttk.Combobox(frame, values=["Mean (μ)", "Variance (σ²)", "Proportion (p)"], state="readonly", width=18)
        widgets['param'].grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        widgets['param'].set("Mean (μ)")
        widgets['param'].bind("<<ComboboxSelected>>", lambda e: self.update_ht_inputs(widgets))

        # Input Frame
        input_frame = ttk.LabelFrame(frame, text="Inputs & Hypotheses")
        input_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky='ew')
        widgets['input_frame'] = input_frame

        # Dynamic Input Area
        self.update_ht_inputs(widgets, initial=True)

        # Calculate Button
        calc_button = ttk.Button(frame, text="Perform Hypothesis Test", command=lambda: self.calculate_ht(widgets))
        calc_button.grid(row=2, column=0, columnspan=4, pady=15)

    def update_ht_inputs(self, widgets, initial=False):
        # This function remains the same as the previous version
        param = widgets['param'].get()
        frame = widgets['input_frame']
        for widget in frame.winfo_children():
            widget.destroy() # Clear previous

        # Common Alpha and Alternative
        ttk.Label(frame, text="Alpha (α):").grid(row=0, column=2, padx=5, pady=3, sticky='w')
        widgets['alpha'] = ttk.Entry(frame, width=10)
        widgets['alpha'].grid(row=0, column=3, padx=5, pady=3, sticky='w')
        widgets['alpha'].insert(0, "0.05")

        ttk.Label(frame, text="Alternative (H₁):").grid(row=1, column=2, padx=5, pady=3, sticky='w')
        widgets['alt'] = ttk.Combobox(frame, values=["two-sided (≠)", "less (<)", "greater (>)"], state="readonly", width=15)
        widgets['alt'].grid(row=1, column=3, padx=5, pady=3, sticky='ew')
        widgets['alt'].set("two-sided (≠)")

        # Parameter specific inputs (adjust pady=3)
        if param == "Mean (μ)":
            ttk.Label(frame, text="Method:").grid(row=0, column=0, padx=5, pady=3, sticky='w')
            widgets['mean_method_var'] = tk.StringVar(value="Z")
            ttk.Radiobutton(frame, text="Z (σ known or n>30)", variable=widgets['mean_method_var'], value="Z", command=lambda: self.toggle_sigma_s_ht(widgets)).grid(row=1, column=0, columnspan=2, padx=5, sticky='w')
            ttk.Radiobutton(frame, text="t (σ unknown, n≤30)", variable=widgets['mean_method_var'], value="t", command=lambda: self.toggle_sigma_s_ht(widgets)).grid(row=2, column=0, columnspan=2, padx=5, sticky='w')

            ttk.Label(frame, text="Hypothesized μ₀:").grid(row=3, column=0, padx=5, pady=3, sticky='w')
            widgets['mean_mu0'] = ttk.Entry(frame, width=10)
            widgets['mean_mu0'].grid(row=3, column=1, padx=5, pady=3, sticky='w')
            ttk.Label(frame, text="Sample Mean (x̄):").grid(row=4, column=0, padx=5, pady=3, sticky='w')
            widgets['mean_xbar'] = ttk.Entry(frame, width=10)
            widgets['mean_xbar'].grid(row=4, column=1, padx=5, pady=3, sticky='w')
            ttk.Label(frame, text="Sample Size (n):").grid(row=5, column=0, padx=5, pady=3, sticky='w')
            widgets['mean_n'] = ttk.Entry(frame, width=10)
            widgets['mean_n'].grid(row=5, column=1, padx=5, pady=3, sticky='w')

             # Sigma / S input area
            widgets['sigma_label'] = ttk.Label(frame, text="Pop. Std Dev (σ):")
            widgets['sigma_entry'] = ttk.Entry(frame, width=10)
            widgets['s_label'] = ttk.Label(frame, text="Sample Std Dev (s):")
            widgets['s_entry'] = ttk.Entry(frame, width=10)
            self.toggle_sigma_s_ht(widgets) # Show initial Z input

        elif param == "Variance (σ²)":
            ttk.Label(frame, text="Hypothesized σ₀²:").grid(row=2, column=0, padx=5, pady=3, sticky='w')
            widgets['var_sigma0_sq'] = ttk.Entry(frame, width=10)
            widgets['var_sigma0_sq'].grid(row=2, column=1, padx=5, pady=3, sticky='w')
            ttk.Label(frame, text="Sample Variance (s²):").grid(row=3, column=0, padx=5, pady=3, sticky='w')
            widgets['var_s2'] = ttk.Entry(frame, width=10)
            widgets['var_s2'].grid(row=3, column=1, padx=5, pady=3, sticky='w')
            ttk.Label(frame, text="Sample Size (n):").grid(row=4, column=0, padx=5, pady=3, sticky='w')
            widgets['var_n'] = ttk.Entry(frame, width=10)
            widgets['var_n'].grid(row=4, column=1, padx=5, pady=3, sticky='w')

        elif param == "Proportion (p)":
            ttk.Label(frame, text="Hypothesized p₀:").grid(row=2, column=0, padx=5, pady=3, sticky='w')
            widgets['prop_p0'] = ttk.Entry(frame, width=10)
            widgets['prop_p0'].grid(row=2, column=1, padx=5, pady=3, sticky='w')
            ttk.Label(frame, text="Successes (x):").grid(row=3, column=0, padx=5, pady=3, sticky='w')
            widgets['prop_x'] = ttk.Entry(frame, width=10)
            widgets['prop_x'].grid(row=3, column=1, padx=5, pady=3, sticky='w')
            ttk.Label(frame, text="Sample Size (n):").grid(row=4, column=0, padx=5, pady=3, sticky='w')
            widgets['prop_n'] = ttk.Entry(frame, width=10)
            widgets['prop_n'].grid(row=4, column=1, padx=5, pady=3, sticky='w')

    def toggle_sigma_s_ht(self, widgets, start_row=6): # Add start_row
         frame = widgets['input_frame']
         method = widgets['mean_method_var'].get()
         # Use pady=3 and start_row
         if method == "Z":
             widgets['s_label'].grid_remove()
             widgets['s_entry'].grid_remove()
             widgets['sigma_label'].grid(row=start_row, column=0, padx=5, pady=3, sticky='w')
             widgets['sigma_entry'].grid(row=start_row, column=1, padx=5, pady=3, sticky='w')
         else: # method == "t"
             widgets['sigma_label'].grid_remove()
             widgets['sigma_entry'].grid_remove()
             widgets['s_label'].grid(row=start_row, column=0, padx=5, pady=3, sticky='w')
             widgets['s_entry'].grid(row=start_row, column=1, padx=5, pady=3, sticky='w')

    def calculate_ht(self, widgets):
        # This function remains the same as the previous version
        try:
            param = widgets['param'].get()
            alpha = self.get_float(widgets['alpha'], -1)
            alt_full = widgets['alt'].get()
            alt_map = {"two-sided (≠)": "two-sided", "less (<)": "less", "greater (>)": "greater"}
            alternative = alt_map.get(alt_full, "two-sided")
            result = "Error: Invalid inputs."

            if param == "Mean (μ)":
                method = widgets['mean_method_var'].get()
                mu0 = self.get_float(widgets['mean_mu0'])
                mean = self.get_float(widgets['mean_xbar'])
                n = self.get_int(widgets['mean_n'])
                if mu0 is None or mean is None or n is None: raise ValueError("μ₀, x̄, and n required.")

                if method == "Z":
                    sigma = self.get_float(widgets['sigma_entry'])
                    if sigma is None: raise ValueError("Sigma required for Z test.")
                    result = ht_mean_z(mean, mu0, sigma, n, alpha, alternative)
                else: # method == "t"
                    s = self.get_float(widgets['s_entry'])
                    if s is None: raise ValueError("Sample Std Dev (s) required for t test.")
                    result = ht_mean_t(mean, mu0, s, n, alpha, alternative)

            elif param == "Variance (σ²)":
                 sigma0_sq = self.get_float(widgets['var_sigma0_sq'])
                 s2 = self.get_float(widgets['var_s2'])
                 n = self.get_int(widgets['var_n'])
                 if sigma0_sq is None or s2 is None or n is None: raise ValueError("σ₀², s², and n required.")
                 result = ht_variance(s2, sigma0_sq, n, alpha, alternative)

            elif param == "Proportion (p)":
                 p0 = self.get_float(widgets['prop_p0'])
                 x = self.get_int(widgets['prop_x'])
                 n = self.get_int(widgets['prop_n'])
                 if p0 is None or x is None or n is None: raise ValueError("p₀, x, and n required.")
                 result = ht_proportion(x, n, p0, alpha, alternative)

            self.display_results(result)

        except Exception as e:
             messagebox.showerror("Input Error", f"Could not perform calculation.\nCheck inputs.\nDetails: {e}")
             self.display_results(f"Error during calculation: {e}")

    def create_ss_tab(self):
        # This function remains the same as the previous version
        frame = self.tab_ss
        widgets = {}

        # Parameter Selection
        ttk.Label(frame, text="Parameter for CI:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        widgets['param'] = ttk.Combobox(frame, values=["Mean (μ)", "Proportion (p)"], state="readonly", width=18)
        widgets['param'].grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        widgets['param'].set("Mean (μ)")
        widgets['param'].bind("<<ComboboxSelected>>", lambda e: self.update_ss_inputs(widgets))

        # Input Frame
        input_frame = ttk.LabelFrame(frame, text="Inputs")
        input_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky='ew')
        widgets['input_frame'] = input_frame

        self.update_ss_inputs(widgets, initial=True)

        # Calculate Button
        calc_button = ttk.Button(frame, text="Calculate Required Sample Size", command=lambda: self.calculate_ss(widgets))
        calc_button.grid(row=2, column=0, columnspan=4, pady=15) # Increased pady

    def update_ss_inputs(self, widgets, initial=False):
        # This function remains the same as the previous version
        param = widgets['param'].get()
        frame = widgets['input_frame']
        for widget in frame.winfo_children(): widget.destroy() # Clear

        # Common Inputs (adjust pady=3)
        ttk.Label(frame, text="Desired Margin of Error (E):").grid(row=0, column=0, padx=5, pady=3, sticky='w')
        widgets['error'] = ttk.Entry(frame, width=10)
        widgets['error'].grid(row=0, column=1, padx=5, pady=3, sticky='w')
        ttk.Label(frame, text="Alpha (α):").grid(row=1, column=0, padx=5, pady=3, sticky='w')
        widgets['alpha'] = ttk.Entry(frame, width=10)
        widgets['alpha'].grid(row=1, column=1, padx=5, pady=3, sticky='w')
        widgets['alpha'].insert(0, "0.05")

        # Parameter specific (adjust pady=3)
        if param == "Mean (μ)":
             ttk.Label(frame, text="Estimated Pop. Std Dev (σ):").grid(row=2, column=0, padx=5, pady=3, sticky='w')
             widgets['sigma_est'] = ttk.Entry(frame, width=10)
             widgets['sigma_est'].grid(row=2, column=1, padx=5, pady=3, sticky='w')
        elif param == "Proportion (p)":
             ttk.Label(frame, text="Estimated Proportion (p̂):").grid(row=2, column=0, padx=5, pady=3, sticky='w')
             widgets['p_est'] = ttk.Entry(frame, width=10)
             widgets['p_est'].grid(row=2, column=1, padx=5, pady=3, sticky='w')
             ttk.Label(frame, text="(Leave blank to use 0.5)").grid(row=2, column=2, padx=5, pady=3, sticky='w', columnspan=2)

    def calculate_ss(self, widgets):
         # This function remains the same as the previous version
         try:
            param = widgets['param'].get()
            error = self.get_float(widgets['error'])
            alpha = self.get_float(widgets['alpha'])
            if error is None or alpha is None: raise ValueError("Error (E) and Alpha required.")
            result = "Error: Invalid inputs."

            if param == "Mean (μ)":
                 sigma = self.get_float(widgets['sigma_est'])
                 if sigma is None: raise ValueError("Sigma estimate required.")
                 result = sample_size_mean(sigma, error, alpha)
            elif param == "Proportion (p)":
                 p_est_str = widgets['p_est'].get()
                 p_est = None
                 if p_est_str:
                     p_est = self.get_float(widgets['p_est'])
                     if p_est is None or p_est < 0 or p_est > 1: raise ValueError("Invalid proportion estimate.")
                 # Pass None if blank, function handles it
                 result = sample_size_proportion(p_est, error, alpha)

            self.display_results(result)

         except Exception as e:
             messagebox.showerror("Input Error", f"Could not perform calculation.\nCheck inputs.\nDetails: {e}")
             self.display_results(f"Error during calculation: {e}")

    def create_cv_tab(self):
        # This function remains the same as the previous version
        frame = self.tab_cv
        widgets = {}

        # Distribution Selection
        ttk.Label(frame, text="Distribution:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        widgets['dist'] = ttk.Combobox(frame, values=["Z", "t", "Chi-square (χ²)"], state="readonly", width=18)
        widgets['dist'].grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        widgets['dist'].set("Z")
        widgets['dist'].bind("<<ComboboxSelected>>", lambda e: self.update_cv_inputs(widgets))

        # Input Frame
        input_frame = ttk.LabelFrame(frame, text="Inputs")
        input_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky='ew')
        widgets['input_frame'] = input_frame

        self.update_cv_inputs(widgets, initial=True)

        # Calculate Button
        calc_button = ttk.Button(frame, text="Calculate Critical Value", command=lambda: self.calculate_cv(widgets))
        calc_button.grid(row=2, column=0, columnspan=4, pady=15) # Increased pady

    def update_cv_inputs(self, widgets, initial=False):
        # This function remains the same as the previous version
        dist_type = widgets['dist'].get()
        frame = widgets['input_frame']
        for widget in frame.winfo_children(): widget.destroy() # Clear

        # Adjust pady=3
        ttk.Label(frame, text="Alpha (α) / Tail Area:").grid(row=0, column=0, padx=5, pady=3, sticky='w')
        widgets['alpha'] = ttk.Entry(frame, width=10)
        widgets['alpha'].grid(row=0, column=1, padx=5, pady=3, sticky='w')
        widgets['alpha'].insert(0, "0.05")

        ttk.Label(frame, text="Tail Type:").grid(row=1, column=0, padx=5, pady=3, sticky='w')
        tail_options = []
        df_state = tk.DISABLED # Default df to disabled

        if dist_type == "Z":
            tail_options = ["Two-tailed (±Zα/2)", "Upper tail (Zα)", "Lower tail (-Zα)"]
        elif dist_type == "t":
             tail_options = ["Two-tailed (±tα/2)", "Upper tail (tα)", "Lower tail (-tα)"]
             df_state = tk.NORMAL # Enable df for t
        elif dist_type == "Chi-square (χ²)":
             tail_options = ["Upper tail (χ²α)", "Lower tail (χ²(1-α))", "Upper tail CI (χ²α/2)", "Lower tail CI (χ²(1-α/2))"]
             df_state = tk.NORMAL # Enable df for Chi2

        widgets['tail'] = ttk.Combobox(frame, values=tail_options, state="readonly", width=25)
        widgets['tail'].grid(row=1, column=1, padx=5, pady=3, sticky='ew')
        if tail_options: widgets['tail'].current(0)

        # DF Entry (always grid, but state changes)
        widgets['df_label'] = ttk.Label(frame, text="df:")
        widgets['df_entry'] = ttk.Entry(frame, width=10, state=df_state)
        widgets['df_label'].grid(row=2, column=0, padx=5, pady=3, sticky='w')
        widgets['df_entry'].grid(row=2, column=1, padx=5, pady=3, sticky='w')

    def calculate_cv(self, widgets):
        # This function remains the same as the previous version
        try:
            dist = widgets['dist'].get()
            tail = widgets['tail'].get()
            alpha = self.get_float(widgets['alpha'])
            df = None
            if dist in ["t", "Chi-square (χ²)"]:
                 df = self.get_int(widgets['df_entry'])
                 if df is None or df <= 0: raise ValueError("Valid df (>0) required.")

            if alpha is None: raise ValueError("Alpha required.")

            result = calculate_critical_value(dist, tail, alpha, df)
            self.display_results(result)

        except Exception as e:
             messagebox.showerror("Input Error", f"Could not perform calculation.\nCheck inputs.\nDetails: {e}")
             self.display_results(f"Error during calculation: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = StatsCalculatorGUI(root)
    root.mainloop()