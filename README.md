# PHJD
Numerical methods from: Fitting Financial Phase-Type Jump Diffusions
Prepint available [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6377820).

## Files
`HE_EM.cpp` -  C++ implementation of EM algorithm for hyperexponential (HE) distributions.\
`PHJD.cpp` - C++ implementation of Erlangisation-based routines for numerical evaluation of density, cdf and quantiles for phase-type jump diffusions. Further theoretical details can be found [here](https://link.springer.com/article/10.1007/s41096-024-00209-5).\
`Meixner_example.R` - R code which reproduces results in Example 4.

## Dependencies
This project requires the following R packages:
- matrixdist
- Rcpp
- RcppArmadillo
  
Install them with:
```r
install.packages(c("matrixdist","Rcpp", "RcppArmadillo"))
``` 
## Setup
**1.** Source `HE_EM.cpp`, `PHJD.cpp`.\
**2.** Run `Meixner_example.R`.
