# Single Sample Stats Calculator GUI

The **Single Sample Stats Calculator GUI** is Built with Tkinter and powered by SciPy, allowing you to compute:

- **Confidence Intervals** (for means, variances, and proportions)
- **Hypothesis Tests** (for means, variances, and proportions)
- **Sample Size Estimates** (for mean and proportion calculations)
- **Critical Value Calculations** for Z, t, and Chi-square distributions

The application uses a tabbed interface to separate each functionality into its own section, making it easy to navigate through different types of statistical analysis.

## Features

- **Confidence Interval Calculation:**  
  - Mean (using both Z (when population standard deviation is known or sample size > 30) and t methods)
  - Variance (using the Chi-square distribution)
  - Proportion (using the normal approximation/Wald method)

- **Hypothesis Testing:**  
  - Mean testing with Z or t methods
  - Variance testing (Chi-square based)
  - Proportion testing (using a Z-test)

- **Sample Size Determination:**  
  - Compute the required sample size for mean-based estimates given a desired margin of error
  - Determine the required sample size for proportion-based estimates

- **Critical Value Calculation:**  
  - Quickly calculate critical values for Z, t, and Chi-square distributions based on your selected tail criteria

- **User-Friendly GUI:**  
  - Clean, intuitive interface built with Tkinter’s Notebook widget
  - Dynamic input fields that update based on the selected parameter and method
  - Clear, formatted output with formulas and computed values

## Requirements

- **Python 3**  
- **Tkinter** – included with most Python installations  
- **SciPy** – used for statistical computations  

To install SciPy, run:

```bash
pip install scipy
```

## Installation and Running

1. **Clone the repository or download the file:**

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd your-repo-name
    ```

3. **Run the application:**

    ```bash
    python stats_calc.py
    ```

The GUI window will open, allowing you to select different tabs for Confidence Intervals, Hypothesis Tests, Sample Size calculations, and Critical Value calculations.

## Usage

### Confidence Intervals

- **Select the Parameter:** Choose between Mean, Variance, or Proportion.
- **Specify the Interval Type:** Two-sided, Lower Bound, or Upper Bound.
- **Input Values:** Enter values for the sample statistics, sample size, and significance level (α).
- **Calculate:** Click the "Calculate CI" button to display the confidence interval along with the margin of error, critical values, and relevant formulas.

### Hypothesis Tests

- **Select the Parameter:** Choose between Mean, Variance, or Proportion.
- **Enter Hypotheses and Inputs:** Provide the hypothesized parameter value, sample statistics, and sample size.
- **Choose the Alternative Hypothesis:** Two-sided (≠), less (<), or greater (>).
- **Test:** Click the "Perform Hypothesis Test" button to see the test statistic, p-value, critical values, and decision rule.

### Sample Size Determination

- **Select the Parameter:** Choose between Mean (using population standard deviation) or Proportion.
- **Enter Required Margin of Error and Confidence Level:** Also provide an estimated value (e.g., σ for mean or p̂ for proportion; leave blank for p̂ to default to 0.5).
- **Calculate:** Click the "Calculate Required Sample Size" button to receive the calculated sample size along with the computation details.

### Critical Values

- **Select the Distribution:** Options include Z, t, or Chi-square.
- **Choose the Tail Type:** Based on your hypothesis or confidence interval needs.
- **Input Alpha and Degrees of Freedom (if required):**
- **Compute:** Click the "Calculate Critical Value" button to display the computed value and the formula used.

## Contributing

Contributions to the project are welcome! Feel free to fork the repository, submit issues, or open pull requests for improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- **Python & Tkinter:** For providing the foundation for our GUI.
- **SciPy:** For the robust statistical functions that power the calculations.
