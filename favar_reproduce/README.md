# FAVAR Model - Python Implementation

This project provides a simple Python implementation of the Factor-Augmented Vector Autoregressive (FAVAR) model. We refer the original RATS code by LIN Yingxin.

## Overview

The FAVAR model, introduced by Bernanke et al. (2005), combines factor models with Vector Autoregressive (VAR) models to analyze the effects of monetary policy using large datasets of macroeconomic variables.

## Required Data Files

The implementation expects two Excel files in the parent directory:

- `xdata.XLSX`: Large dataset with 119 macroeconomic variables
- `ydata.XLSX`: Observable factors (e.g., Federal Funds Rate)

### Jupyter Notebook Demo

Open `FAVAR_Demo.ipynb` for a comprehensive demonstration with visualizations.

## References

1. Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the effects of monetary policy: a factor-augmented vector autoregressive (FAVAR) approach. *The Quarterly Journal of Economics*, 120(1), 387-422.
2. Original RATS implementation by LIN Yingxin, Central University of Finance and Economics (CUFE). https://github.com/lyx66/Factor-augmented-vector-autoregressive-FAVAR-WINRATS-code-package-
