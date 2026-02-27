// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <vector>
#include <limits>

double HE_density(double x,
                  arma::vec alpha,
                  arma::vec mu){
  
  int p = alpha.size();
  double val = 0;
  for(int i = 0; i<p; ++i){
    val = val + alpha[i] * mu[i] * exp(-mu[i] * x);
  }
  return val;
}

double EN_i(int i,
            arma::vec x,
            arma::vec w,
            arma::vec alpha,
            arma::vec mu){
  
  int N = x.size();
  
  auto E_i = [&](int n) {
    double val = (alpha[i] * mu[i] * exp(-mu[i] * x[n])) / 
                  HE_density(x[n], alpha, mu);
    return val * w[n];
  };
  
  double E = 0;
  for(int n = 0; n<N; ++n){
    E = E + E_i(n);
  }
  return E;
}  

double ES_i(int i,
            arma::vec x,
            arma::vec w,
            arma::vec alpha,
            arma::vec mu){
  
  int N = x.size();
  
  auto S_i = [&](int n) {
    double val = (x[n] * alpha[i] * mu[i] * exp(-mu[i] * x[n])) / 
                  HE_density(x[n], alpha, mu);
    return val * w[n];
  };
  
  double S = 0;
  for(int n = 0; n<N; ++n){
    S = S + S_i(n);
  }
  return S;
}  

// [[Rcpp::export]]
void EM_HE(arma::vec x,
            arma::vec w,
            arma::vec &alpha,
            arma::vec &mu){
  
  int N = x.size();
  int p = alpha.size();
  
  arma::mat E_step_vals(N,p);
  for(int n = 0; n<N; ++n){
    double HE_density_n = HE_density(x[n], alpha, mu);
    double x_n = x[n];
    for(int i = 0; i<p; ++i){
      E_step_vals(n,i) = w[n] * (alpha[i] * mu[i] * exp(-mu[i] * x_n)) / HE_density_n;
    }
  }
  
  std::vector<double> EN(p);
  std::vector<double> ES(p);
  
  arma::vec alpha_update(p);
  arma::vec mu_update(p);
  double w_sum = sum(w);
  
  for (int i = 0; i < p; i++) {
    EN[i] = sum(E_step_vals.col(i));
    ES[i] = arma::dot(x, E_step_vals.col(i));
  
    alpha_update[i] = EN[i] / w_sum;
    mu_update[i] = EN[i] / ES[i];
  }
  
  alpha = alpha_update;
  mu = mu_update;

}
