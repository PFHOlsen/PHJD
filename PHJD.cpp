// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <vector>
#include <limits>

// Matrix exponential function and related functions are from 
// https://github.com/martinbladt/matrixdist_1.0

int pow2k(int k){
  return static_cast<int>(std::pow(2, k));
}

arma::vec delta0_fun(double delta, int p){
  arma::vec delta0(p+1);
  delta0[0] = delta;
  return delta0;
}

arma::vec e(int p){
  arma::vec e_vec(p);
  e_vec.ones();
  return e_vec;
}

arma::vec exit_vec(arma::mat A){
  int p = A.n_cols;
  arma::vec ev = (A * (-1)) * e(p);
  return ev;
}

void assign_submatrix(arma::mat &A, 
                      const arma::mat& submatrix, 
                      int i, 
                      int j) {
  
  int block_size = submatrix.n_cols;  
  
  int row_start = (i - 1) * block_size;
  int col_start = (j - 1) * block_size;
  
  A.submat(row_start, col_start,
           row_start + block_size - 1,
           col_start + block_size - 1) = submatrix;
}

void assign_subvector(arma::rowvec &v,
                      const arma::rowvec &subvector,
                      int i) {
  int q = subvector.n_cols;
  int start = (i - 1) * q;
  v.cols(start, start + q - 1) = subvector;
}

arma::rowvec stack_rowvec(const std::vector<arma::rowvec>& vecs) {
  int q = vecs.size();
  
  int p1 = vecs[0].size();  
  int total_length = q * p1;
  
  arma::rowvec stacked_vec(total_length);
  
  for (int i = 0; i < q; ++i) {
    assign_subvector(stacked_vec, vecs[i], i + 1);
  }
  return stacked_vec;
}

arma::rowvec extract_subvector(const arma::rowvec &v,
                               int i,
                               int p) {
  int start = (i - 1) * p;
  return v.cols(start, start + p - 1);
}

arma::mat extract_submatrix(arma::mat A, 
                            int i, 
                            int j, 
                            int block_size) {
  
  int row_start = (i - 1) * block_size;
  int col_start = (j - 1) * block_size;
  
  return A.submat(row_start, col_start,
                  row_start + block_size - 1,
                  col_start + block_size - 1);
}

// Numeric integration ////

double inf_norm(arma::mat A) {
  double value{0.0};
  
  for (int i{0}; i < A.n_rows; ++i) {
    double row_sum{0.0};
    for (int j{0}; j < A.n_cols; ++j) {
      row_sum += fabs(A(i,j));
    }
    value = std::max(value, row_sum);
  }
  return value;
}

double inf_norm_rowvec(const arma::rowvec& v) {
  double sum = 0.0;
  for (int j = 0; j < v.n_cols; ++j) {
    sum += std::fabs(v[j]);
  }
  return sum;
}

arma::mat resolvent_noinv(double s, arma::mat S){
  
  int p = S.n_rows;
  
  arma::vec s_vec(p);
  s_vec.fill(s);
  
  arma::mat sId(p,p, arma::fill::zeros);
  sId.diag() = s_vec; 
  arma::mat Res = sId - S;
  
  return Res;
  
}

arma::mat resolvent(double s, arma::mat S){
  
  int p = S.n_rows;
  
  arma::vec s_vec(p);
  s_vec.fill(s);
  
  arma::mat sId(p,p, arma::fill::zeros);
  sId.diag() = s_vec; 
  arma::mat Res = inv(sId - S);
  
  return Res;
  
}

arma::mat matrix_exponential(arma::mat A) {
  const int q{6};
  
  arma::mat expm(A.n_rows,A.n_cols);
  
  double a_norm{inf_norm(A)};
  
  int ee{static_cast<int>(log2(a_norm)) + 1};
  
  int s{std::max(0, ee + 1)};
  
  double t{1.0 / pow(2.0, s)};
  
  arma::mat a2 = A * t;
  arma::mat x = a2;
  
  double c{0.5};
  
  expm.eye(size(A));
  expm = expm + (a2 * c);
  
  arma::mat d;
  d.eye(size(A));
  d = (d + a2 * (-c));
  
  int p{1};
  
  for (int k{2}; k <= q; ++k) {
    c = c * static_cast<double>(q - k + 1) / static_cast<double>(k * (2 * q - k + 1));
    x = (a2 * x);
    expm = (x * c) + expm;
    if (p) {
      d = (x * c) + d;
    }
    else {
      d = (x * (-c)) + d;
    }
    p = !p;
  }
  expm = inv(d) * expm;
  for (int k = 1; k <= s; ++k) {
    expm = expm * expm;
  }
  return(expm);
}


double find_T_max(double eps, const arma::rowvec& alpha_dens, const arma::mat& T_dens, const arma::vec& t_dens){
  
  arma::mat expM_1 = matrix_exponential(T_dens);
  arma::mat expM_t = expM_1;
  
  auto dens = [&](const arma::mat&  M) {
    return arma::as_scalar(alpha_dens * M * t_dens);
  };
  
  double T_max = 1;
  while(dens(expM_t)>eps){
    expM_t = expM_t * expM_1;
    T_max = T_max + 0.01;
  }
  return T_max;
}

std::vector<arma::mat> expM_grid_fun(const arma::mat& M,
                                     double s, double t, int n){
  double h = (t-s) / n;
  std::vector<arma::mat> expM_vec;
  expM_vec.reserve(n + 1);
  arma::mat expM_idx = matrix_exponential(M * s);
  arma::mat expM_inc = matrix_exponential(M * h);
  
  expM_vec.push_back(expM_idx);
  for (int i = 1; i <= n; ++i) {
    expM_idx = expM_idx * expM_inc; 
    expM_vec.push_back(expM_idx);
  }
  return expM_vec;
}

std::vector<double> dens_grid_fun(const arma::rowvec& alpha_dens, const arma::mat& T_dens, const arma::vec& t_dens,
                                  double s, double t, int n){
  double h = (t-s) / n;
  std::vector<double> dens_vec(n+1);
  arma::mat expM_idx = matrix_exponential(T_dens * s);
  arma::mat expM_inc = matrix_exponential(T_dens * h);
  
  auto dens = [&](const arma::mat&  M) {
    return arma::as_scalar(alpha_dens * M * t_dens);
  };
  
  dens_vec[0] = dens(expM_idx);
  for (int i = 1; i <= n; ++i) {
    expM_idx = expM_idx * expM_inc; 
    dens_vec[i] = dens(expM_idx);
  }
  return dens_vec;
}

std::vector<arma::vec> expM_vec_grid_fun(const arma::mat& M,
                                         const arma::vec& v,
                                         double s, double t, int n){
  double h = (t-s) / n;
  std::vector<arma::vec> expM_vec;
  expM_vec.reserve(n + 1);
  arma::vec expM_idx = matrix_exponential(M * s) * v;
  arma::mat expM_inc = matrix_exponential(M * h);
  
  expM_vec.push_back(expM_idx);
  for (int i = 1; i <= n; ++i) {
    expM_idx = expM_inc * expM_idx; 
    expM_vec.push_back(expM_idx);
  }
  return expM_vec;
}

std::vector<arma::rowvec> expM_rowvec_grid_fun(const arma::mat& M,
                                               const arma::rowvec& a,
                                               double s, double t, int n){
  double h = (t-s) / n;
  std::vector<arma::rowvec> expM_rowvec;
  expM_rowvec.reserve(n + 1);
  arma::rowvec expM_idx = a * matrix_exponential(M * s);
  arma::mat expM_inc = matrix_exponential(M * h);
  
  expM_rowvec.push_back(expM_idx);
  for (int i = 1; i <= n; ++i) {
    expM_idx = arma::rowvec(expM_idx * expM_inc); 
    expM_rowvec.push_back(expM_idx);
  }
  return expM_rowvec;
}

arma::mat expG_int_trapez_subgrid(const std::vector<arma::mat>& expG_grid,
                                  const std::vector<double>& dens_grid,
                                  int stride, double h_fine){
  int N = expG_grid.size() - 1;
  int p = expG_grid[0].n_rows;
  
  arma::mat I = arma::zeros(p, p);
  
  I += 0.5 * dens_grid[0] * expG_grid[0];
  for (int i = stride; i < N; i += stride) {
    I += dens_grid[i] * expG_grid[i];
  }
  I += 0.5 * dens_grid[N] * expG_grid[N];
  
  double h_sub = h_fine * stride;
  return h_sub * I;
}

arma::vec expG_vec_int_trapez_subgrid(const std::vector<arma::vec>& expG_vec_grid,
                                      const std::vector<double>& dens_grid,
                                      int stride, double h_fine){
  int N = expG_vec_grid.size() - 1;
  int p = expG_vec_grid[0].size();
  
  arma::vec I = arma::zeros<arma::vec>(p);
  
  I += 0.5 * dens_grid[0] * expG_vec_grid[0];
  for (int i = stride; i < N; i += stride) {
    I += dens_grid[i] * expG_vec_grid[i];
  }
  I += 0.5 * dens_grid[N] * expG_vec_grid[N];
  
  double h_sub = h_fine * stride;
  return h_sub * I;
}

arma::rowvec expG_rowvec_int_trapez_subgrid(const std::vector<arma::rowvec>& expG_rowvec_grid,
                                            const std::vector<double>& dens_grid,
                                            int stride, double h_fine){
  int N = expG_rowvec_grid.size() - 1;
  int p = expG_rowvec_grid[0].size();
  
  arma::rowvec I = arma::zeros<arma::rowvec>(p);
  
  I += 0.5 * dens_grid[0] * expG_rowvec_grid[0];
  for (int i = stride; i < N; i += stride) {
    I += dens_grid[i] * expG_rowvec_grid[i];
  }
  I += 0.5 * dens_grid[N] * expG_rowvec_grid[N];
  
  double h_sub = h_fine * stride;
  return h_sub * I;
}

arma::mat expG_int(const arma::mat& G,
                   const arma::rowvec& alpha_dens,
                   const arma::mat& T_dens,
                   const arma::vec& t_dens,
                   int n_base = 500,
                   int n_levels = 5) {
  int p = G.n_cols;
  
  double eps = 1e-12;
  double T_max = find_T_max(eps, alpha_dens, T_dens, t_dens);
  
  int n_fine = n_base * pow2k(n_levels - 1);
  double h_fine = T_max / n_fine;
  
  std::vector<arma::mat> expG_grid = expM_grid_fun(G, 0.0, T_max, n_fine);
  std::vector<double> dens_grid = dens_grid_fun(alpha_dens, T_dens, t_dens, 0.0, T_max, n_fine);
  
  std::vector<std::vector<arma::mat>> R(n_levels);
  for (int k = 0; k < n_levels; ++k)
    R[k].resize(k + 1, arma::zeros<arma::mat>(p, p));
  
  for (int k = 0; k < n_levels; ++k) {
    int stride = pow2k(n_levels - 1 - k); 
    R[k][0] = expG_int_trapez_subgrid(expG_grid, dens_grid, stride, h_fine);
  }
  
  for (int m = 1; m < n_levels; ++m) {
    for (int k = m; k < n_levels; ++k) {
      R[k][m] = R[k][m - 1] + (R[k][m - 1] - R[k - 1][m - 1]) / (std::pow(4, m) - 1);
    }
  }
  
  return R[n_levels - 1][n_levels - 1];
}

arma::mat expG_int_HE(const arma::mat& G,
                      const arma::rowvec& alpha_dens,
                      const arma::mat& T_dens) 
{
  int p = G.n_cols;
  arma::mat res(p,p, arma::fill::zeros);
  arma::mat I = arma::eye(p,p);
  arma::vec rates = T_dens.diag();
  
  for(int i = 0; i < rates.size(); i++){
    res = res + alpha_dens[i] * rates[i] * inv(G + I * rates[i]); // rates are negative
  }
  return res;
}

arma::vec expG_vec_int(const arma::mat& G,
                       const arma::vec v,
                       const arma::rowvec& alpha_dens,
                       const arma::mat& T_dens,
                       const arma::vec& t_dens,
                       int n_base = 500,
                       int n_levels = 5) {
  int p = G.n_cols;
  
  double eps = 1e-12;
  double T_max = find_T_max(eps, alpha_dens, T_dens, t_dens);
  
  int n_fine = n_base * pow2k(n_levels - 1);
  double h_fine = T_max / n_fine;
  
  std::vector<arma::vec> expG_vec_grid = expM_vec_grid_fun(G, v, 0.0, T_max, n_fine);
  std::vector<double> dens_grid = dens_grid_fun(alpha_dens, T_dens, t_dens, 0.0, T_max, n_fine);
  
  std::vector<std::vector<arma::vec>> R(n_levels);
  for (int k = 0; k < n_levels; ++k)
    R[k].resize(k + 1, arma::zeros<arma::vec>(p));
  
  for (int k = 0; k < n_levels; ++k) {
    int stride = pow2k(n_levels - 1 - k); 
    R[k][0] = expG_vec_int_trapez_subgrid(expG_vec_grid, dens_grid, stride, h_fine);
  }
  
  for (int m = 1; m < n_levels; ++m) {
    for (int k = m; k < n_levels; ++k) {
      R[k][m] = R[k][m - 1] + (R[k][m - 1] - R[k - 1][m - 1]) / (std::pow(4, m) - 1);
    }
  }

  return R[n_levels - 1][n_levels - 1];
}

arma::rowvec expG_rowvec_int(const arma::mat& G,
                             const arma::rowvec a,
                             const arma::rowvec& alpha_dens,
                             const arma::mat& T_dens,
                             const arma::vec& t_dens,
                             int n_base = 500,
                             int n_levels = 5) {
  int p = G.n_cols;
  
  double eps = 1e-12;
  double T_max = find_T_max(eps, alpha_dens, T_dens, t_dens);
  
  int n_fine = n_base * pow2k(n_levels - 1);
  double h_fine = T_max / n_fine;
  
  std::vector<arma::rowvec> expG_rowvec_grid = expM_rowvec_grid_fun(G, a, 0.0, T_max, n_fine);
  std::vector<double> dens_grid = dens_grid_fun(alpha_dens, T_dens, t_dens, 0.0, T_max, n_fine);
  
  std::vector<std::vector<arma::rowvec>> R(n_levels);
  for (int k = 0; k < n_levels; ++k)
    R[k].resize(k + 1, arma::zeros<arma::rowvec>(p));
  
  for (int k = 0; k < n_levels; ++k) {
    int stride = pow2k(n_levels - 1 - k); 
    R[k][0] = expG_rowvec_int_trapez_subgrid(expG_rowvec_grid, dens_grid, stride, h_fine);
  }
  
  for (int m = 1; m < n_levels; ++m) {
    for (int k = m; k < n_levels; ++k) {
      R[k][m] = R[k][m - 1] + (R[k][m - 1] - R[k - 1][m - 1]) / (std::pow(4, m) - 1);
    }
  }
  
  return R[n_levels - 1][n_levels - 1];
}

arma::mat upper_toeplitz(const std::vector<arma::mat>& mat_vec) {
  int n = mat_vec.size();
  arma::mat first = mat_vec[0];
  int block_size = first.n_rows;
  arma::mat result = arma::zeros(block_size * n, block_size * n);
  
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      arma::mat block = mat_vec[j - i];
      result.submat(i*block_size, j*block_size,
                    (i+1)*block_size - 1, (j+1)*block_size - 1) = block;
    }
  }
  
  return result;
}

arma::mat G0_fun(arma::rowvec beta, double delta, const arma::mat& T_up) {
  int p = T_up.n_cols;
  arma::vec t_up_vec = exit_vec(T_up);
  double beta_01 = beta[0];
  arma::rowvec beta_1p1 = beta.cols(1, p);
  arma::mat G0(p+1, p+1);
  G0(0,0) = -delta * (1.0-beta_01);
  G0(arma::span(0, 0), arma::span(1, p)) = delta * beta_1p1;
  G0(arma::span(1, p), arma::span(0, 0)) = t_up_vec;
  G0(arma::span(1, p), arma::span(1, p)) = T_up;
  return G0;
}

arma::mat G_fun(arma::rowvec beta, int p, int q, double delta, const arma::mat& T_up){
  arma::vec delta0 = delta0_fun(delta, p);
  arma::mat G0 = G0_fun(beta, delta, T_up);
  std::vector<arma::mat> UT_vec(q, arma::mat(p+1, p+1));
  UT_vec[0] = G0;
  for (int i = 1; i < q; ++i) {
    UT_vec[i] = delta0 * extract_subvector(beta, i+1, p+1);
  }
  arma::mat UT = upper_toeplitz(UT_vec);
  return UT;
}

bool isHE(arma::mat G){
  int p = G.n_rows;
  
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      if(i == j){continue;}
      if(G(i,j) != 0){return false;}
    }  
  }
  return true;
}

std::vector<arma::mat> phi_minus_fun(arma::mat G, int p, int q, arma::rowvec alpha_down, arma::mat T_down){
  arma::mat EGV;
  if(isHE(G)){
    EGV = expG_int_HE(G, alpha_down, T_down);
  } else {
    EGV = expG_int(G, alpha_down, T_down, exit_vec(T_down), 500);
  }
  std::vector<arma::mat> phi_minus_vec(q, arma::mat(p+1,p+1));
  for (int i = 0; i < q; ++i) {
    phi_minus_vec[i] = extract_submatrix(EGV, 1, i + 1, p+1);
  }
  return phi_minus_vec;
}

arma::rowvec psi_fun(arma::rowvec beta,
                     int p, int q,
                     double eta, double a, double sig2,
                     double lambda_up, arma::rowvec alpha_up, arma::mat T_up,
                     double lambda_down, arma::rowvec alpha_down, arma::mat T_down){
  
  arma::rowvec e0(p+1, arma::fill::zeros);
  e0[0] = 1.0;
  
  double lambda_omega = lambda_up + lambda_down + eta;
  double delta = -a/sig2 + pow( pow(a/sig2,2) + 2.0 * lambda_omega / sig2, 0.5 );
  double rho   =  a/sig2 + pow( pow(a/sig2,2) + 2.0 * lambda_omega / sig2, 0.5 );
  double w = rho / lambda_omega;
  
  arma::mat G = G_fun(beta, p, q, delta, T_up);
  
  std::vector<arma::mat> phi_minus = phi_minus_fun(G, p, q, alpha_down, T_down);
  
  std::vector<arma::rowvec> gamma_vec(q, arma::rowvec(p+1));
  
  arma::rowvec alpha0(p + 1);
  alpha0[0] = 0;
  alpha0.subvec(1, p) = alpha_up;
  
  gamma_vec[0] = w * lambda_up * alpha0 +
    w * lambda_down * e0 * phi_minus[0];
  
  if(q>1){
    gamma_vec[1] = w * lambda_down * e0 * phi_minus[1] +
      w * eta * e0;
  }
  if(q>2){
    for (int k = 2; k < q; ++k) {
      gamma_vec[k] = w * lambda_down * e0 * phi_minus[k];
    }
  }
  
  arma::rowvec gamma_stacked = stack_rowvec(gamma_vec);
  
  arma::vec psi_t = solve(
    resolvent_noinv(rho, G).t(), gamma_stacked.t(),
    arma::solve_opts::no_approx + arma::solve_opts::allow_ugly);
  
  arma::rowvec psi = psi_t.t();
  
  return psi;
  
}

// [[Rcpp::export]]
Rcpp::List psi_fixed_point(int p, int q,
                           double eta, double a, double sig2,
                           double lambda_up, arma::rowvec alpha_up,   arma::mat T_up,
                           double lambda_down, arma::rowvec alpha_down, arma::mat T_down){
  
  arma:: rowvec beta(q*(p+1), arma::fill::zeros);
  arma::mat G(q*(p+1), q*(p+1), arma::fill::zeros);
  
  double lambda_omega = lambda_up + lambda_down + eta;
  double delta = -a/sig2 + pow( pow(a/sig2,2) + 2.0 * lambda_omega / sig2, 0.5 );
  
  double inf_norm_old = inf_norm_rowvec(beta);
  double inf_norm_new;
  
  double rele = std::numeric_limits<double>::max();
  int it = 0;
  while(rele > pow(10,-8)){
    it = it + 1;
    
    beta = psi_fun(beta,
                   p, q,
                   eta, a, sig2,
                   lambda_up, alpha_up, T_up,
                   lambda_down, alpha_down, T_down);
    inf_norm_new = inf_norm_rowvec(beta);
    rele = (inf_norm_new - inf_norm_old) / inf_norm_new;
    inf_norm_old = inf_norm_new;
  }
  
  G = G_fun(beta, p, q, delta, T_up);
  
  return Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("G") = G,
    Rcpp::Named("it") = it
  );
}

arma::mat gs_mat_fun(const arma::mat& G, int p, int q) {
  int block = p + 1;
  arma::vec row_sums = -arma::sum(G, 1);
  arma::mat gs_stack(q * block, q, arma::fill::zeros);
  
  for (int s = 0; s < q; ++s) {
    int start = s * block;
    gs_stack.submat(start, s, start + block - 1, s) = row_sums.subvec(start, start + block - 1);
  }
  
  return gs_stack;
}

double cs_fun(const arma::mat& G, const arma::vec& gs){
  arma::vec cs_v = solve(-G, gs, arma::solve_opts::no_approx + arma::solve_opts::allow_ugly);
  return cs_v[0];
}

std::vector<double> cs_vec_fun(const arma::mat& G, const arma::mat& gs_mat){
  int q = gs_mat.n_cols;
  std::vector<double> cs_temp(q);
  for(int s = 0; s < q; ++s){
    cs_temp[s] = cs_fun(G, gs_mat.col(s));
  }
  return cs_temp;
}

arma::mat X_dens_exitvecs_fun(const arma::mat& G_up,
                              const arma::mat& gs_mat_up,
                              const arma::mat& G_down,
                              const arma::mat& gs_mat_down){
  
  int p_Gup   = G_up.n_cols;
  int p_Gdown = G_down.n_cols;
  int q = gs_mat_up.n_cols;
  
  arma::rowvec e0_down(p_Gdown, arma::fill::zeros);
  e0_down[0] = 1.0;
  
  arma::mat X_dens_mat(p_Gup, q);
  for(int s = 0; s < q; s ++){
    X_dens_mat.col(s) = expG_vec_int(G_up, gs_mat_up.col(s), e0_down, G_down, gs_mat_down.col(q-s-1), 1000) ;
  }
  return X_dens_mat;
}

arma::mat X_dens_startvecs_fun(const arma::mat& G_up,
                               const arma::mat& gs_mat_up,
                               const arma::mat& G_down,
                               const arma::mat& gs_mat_down){
  int p_Gup   = G_up.n_cols;
  int p_Gdown = G_down.n_cols;
  int q = gs_mat_up.n_cols;
  
  arma::rowvec e0_up(p_Gup, arma::fill::zeros);
  arma::rowvec e0_down(p_Gdown, arma::fill::zeros);
  e0_up[0] = 1.0;
  e0_down[0] = 1.0;
  
  arma::mat X_dens_mat(q, p_Gdown);
  for(int s = 0; s < q; s ++){
    X_dens_mat.row(s) = expG_rowvec_int(G_down, e0_down, e0_up, G_up, gs_mat_up.col(s), 1000) ;
  }
  return X_dens_mat;
}

// [[Rcpp::export]]
std::vector<double> X_dens(std::vector<double> x ,
                           const arma::mat& G_up,
                           const arma::mat& G_down,
                           int q){
  
  int p_up   = static_cast<int>((G_up.n_cols / q) - 1.0);
  int p_down = static_cast<int>((G_down.n_cols / q) - 1.0);
  
  arma::rowvec e0_up(G_up.n_cols, arma::fill::zeros);
  e0_up[0] = 1.0;

  arma::mat gs_mat_up   = gs_mat_fun(G_up, p_up, q);
  arma::mat gs_mat_down = gs_mat_fun(G_down, p_down, q);
  std::vector<double> cs_vec = cs_vec_fun(G_up, gs_mat_up);
  
  arma::mat X_dens_startvecs = X_dens_startvecs_fun(G_up,
                                                    gs_mat_up,
                                                    G_down,
                                                    gs_mat_down);
  
  arma::mat X_dens_exitvecs = X_dens_exitvecs_fun(G_up,
                                                  gs_mat_up,
                                                  G_down,
                                                  gs_mat_down);
  
  
  auto dens = [&](double z) {
    double val = 0.0;
    if(z >= 0){
      for(int s = 0; s < q; s++){
        val = val + arma::as_scalar(e0_up * matrix_exponential(G_up * z) * X_dens_exitvecs.col(s)) / cs_vec[s];
      } 
    } else {
      for(int s = 0; s < q; s++){
        val = val + arma::as_scalar(X_dens_startvecs.row(s) * matrix_exponential(G_down * (-z)) * gs_mat_down.col(q-s-1)) / cs_vec[s];
      }
    }
    return val;
  };
  
  std::vector<double> dens_vals(x.size());
  for(int i = 0; i < x.size(); i++){
    dens_vals[i] = dens(x[i]);
  }
  return dens_vals;
}

// [[Rcpp::export]]
std::vector<double> X_cdf(std::vector<double> x ,
                          const arma::mat& G_up,
                          const arma::mat& G_down,
                          int q){
  
  int p_up   = static_cast<int>((G_up.n_cols / q) - 1.0);
  int p_down = static_cast<int>((G_down.n_cols / q) - 1.0);
  
  arma::mat I = arma::eye(G_up.n_rows,G_up.n_rows);
  
  arma::rowvec e0_up(G_up.n_cols, arma::fill::zeros);
  e0_up[0] = 1.0;

  arma::mat gs_mat_up   = gs_mat_fun(G_up, p_up, q);
  arma::mat gs_mat_down = gs_mat_fun(G_down, p_down, q);
  std::vector<double> cs_vec = cs_vec_fun(G_up, gs_mat_up);
  
  arma::mat X_dens_startvecs = X_dens_startvecs_fun(G_up,
                                                    gs_mat_up,
                                                    G_down,
                                                    gs_mat_down);
  
  arma::mat X_dens_exitvecs = X_dens_exitvecs_fun(G_up,
                                                  gs_mat_up,
                                                  G_down,
                                                  gs_mat_down);
  
  
  auto cdf = [&](double z) {
    double val = 0.0;
    if(z >= 0){
      for(int s = 0; s < q; s++){
        val = val + 
          arma::as_scalar(X_dens_startvecs.row(s) * inv(-G_down) * gs_mat_down.col(q-s-1)) / cs_vec[s] + 
          arma::as_scalar(e0_up * inv(G_up) * (matrix_exponential(G_up * z)- I) * X_dens_exitvecs.col(s)) / cs_vec[s];
      } 
    } else {
      for(int s = 0; s < q; s++){
        val = val + arma::as_scalar(X_dens_startvecs.row(s) * matrix_exponential(G_down * (-z)) * inv(-G_down) * gs_mat_down.col(q-s-1)) / cs_vec[s];
      }
    }
    return val;
  };
  
  std::vector<double> cdf_vals(x.size());
  for(int i = 0; i < x.size(); i++){
    cdf_vals[i] = cdf(x[i]);
  }
  return cdf_vals;
}

double bisection_invert(const std::function<double(double)>& F,
                        double u,
                        double lower,
                        double upper,
                        double tol,
                        int max_iter) {
  
  double mid; 
  double Fmid;
  
  for (int iter = 0; iter < max_iter; ++iter) {
    mid  = 0.5 * (lower + upper);
    Fmid = F(mid);
    
    if (std::abs(Fmid - u) < tol) {
      return mid;
    }
    
    if (Fmid < u) {
      lower = mid;
    } else {
      upper = mid;
    }
  }
  
  return 0.5 * (lower + upper);
}

// [[Rcpp::export]]
std::vector<double> X_quantile(std::vector<double> u,
                               const arma::mat& G_up,
                               const arma::mat& G_down,
                               int q,
                               double lower,
                               double upper,
                               double tol = 1e-8,
                               int max_iter = 100) {
  
  int p_up   = static_cast<int>((G_up.n_cols / q) - 1.0);
  int p_down = static_cast<int>((G_down.n_cols / q) - 1.0);
  
  arma::mat I = arma::eye(G_up.n_rows, G_up.n_rows);
  
  arma::rowvec e0_up(G_up.n_cols, arma::fill::zeros);
  e0_up[0] = 1.0;
  
  arma::mat gs_mat_up   = gs_mat_fun(G_up, p_up, q);
  arma::mat gs_mat_down = gs_mat_fun(G_down, p_down, q);
  std::vector<double> cs_vec = cs_vec_fun(G_up, gs_mat_up);
  
  arma::mat X_dens_startvecs = X_dens_startvecs_fun(G_up, gs_mat_up, G_down, gs_mat_down);
  
  arma::mat X_dens_exitvecs = X_dens_exitvecs_fun(G_up, gs_mat_up, G_down, gs_mat_down);
  
  arma::mat invGdown = inv(-G_down);
  arma::mat invGup   = inv(G_up);
  
  auto cdf = [&](double z) {
    double val = 0.0;
    if(z >= 0){
      for(int s = 0; s < q; s++){
        val = val + 
          arma::as_scalar(X_dens_startvecs.row(s) * inv(-G_down) * gs_mat_down.col(q-s-1)) / cs_vec[s] + 
          arma::as_scalar(e0_up * inv(G_up) * (matrix_exponential(G_up * z)- I) * X_dens_exitvecs.col(s)) / cs_vec[s];
      } 
    } else {
      for(int s = 0; s < q; s++){
        val = val + arma::as_scalar(X_dens_startvecs.row(s) * matrix_exponential(G_down * (-z)) * inv(-G_down) * gs_mat_down.col(q-s-1)) / cs_vec[s];
      }
    }
    return val;
  };
  
  std::vector<double> x(u.size());
  
  for (int i = 0; i < u.size(); ++i) {
    x[i] = bisection_invert(
      cdf,
      u[i],
       lower,  
       upper,   
       tol,
       max_iter
    );
  }
  
  return x;
}

bool is_equidistant(const std::vector<double>& x, double tol = 1e-12) {
  if (x.size() < 2) return true;
  std::vector<double> diffs(x.size() - 1);
  std::adjacent_difference(x.begin(), x.end(), diffs.begin());
  diffs.erase(diffs.begin());
  double d = diffs[0];
  for (double v : diffs)
    if (std::fabs(v - d) > tol) return false;
    return true;
}

// [[Rcpp::export]]
std::vector<double> X_dens_equi(const std::vector<double>& x,
                                const arma::mat& G_up,
                                const arma::mat& G_down,
                                int q) {
  int n = x.size();
  
  if (!is_equidistant(x, 1e-12)){
    Rcpp::stop("x must be equidistant");
  }
  
  double h = x[1] - x[0];

  int p_up   = static_cast<int>((G_up.n_cols / q) - 1.0);
  int p_down = static_cast<int>((G_down.n_cols / q) - 1.0);
  
  arma::rowvec e0_up(G_up.n_cols, arma::fill::zeros);
  e0_up[0] = 1.0;
  
  arma::mat gs_mat_up   = gs_mat_fun(G_up, p_up, q);
  arma::mat gs_mat_down = gs_mat_fun(G_down, p_down, q);
  std::vector<double> cs_vec = cs_vec_fun(G_up, gs_mat_up);
  
  arma::mat X_dens_startvecs = X_dens_startvecs_fun(G_up, gs_mat_up, G_down, gs_mat_down);
  arma::mat X_dens_exitvecs  = X_dens_exitvecs_fun(G_up, gs_mat_up, G_down, gs_mat_down);
  
  std::vector<double> dens_vals(n, 0.0);

  int idx_pos = 0;
  while (idx_pos < n && x[idx_pos] < 0.0) ++idx_pos;
  
  if (idx_pos < n) {
    arma::mat exp_inc_up = matrix_exponential(G_up * h);
    arma::mat exp_idx_up = matrix_exponential(G_up * x[idx_pos]);
    for (int i = idx_pos; i < n; ++i) {
      double val = 0.0;
      for (int s = 0; s < q; ++s)
        val += arma::as_scalar(e0_up * exp_idx_up * X_dens_exitvecs.col(s)) / cs_vec[s];
      dens_vals[i] = val;
      exp_idx_up = exp_idx_up * exp_inc_up;
    }
  }
  
  if (idx_pos > 0) {
    arma::mat exp_inc_down = matrix_exponential(G_down * h);
    arma::mat exp_idx_down = matrix_exponential(G_down * (-x[idx_pos - 1]));
    for (int i = idx_pos - 1; i >= 0; --i) {
      double val = 0.0;
      for (int s = 0; s < q; ++s)
        val += arma::as_scalar(X_dens_startvecs.row(s) * exp_idx_down * gs_mat_down.col(q - s - 1)) / cs_vec[s];
      dens_vals[i] = val;
      exp_idx_down = exp_idx_down * exp_inc_down;
    }
  }
  
  return dens_vals;
}

// [[Rcpp::export]]
std::vector<double> Romberg_grid(double x_min,
                                 double x_max,
                                 int n_base = 500,
                                 int n_levels = 5) {
  int n_fine = n_base * pow2k(n_levels - 1);  // n_base * 2^(n_levels-1)
  double h = (x_max - x_min) / n_fine;
  
  std::vector<double> grid(n_fine + 1);
  for (int i = 0; i <= n_fine; ++i)
    grid[i] = x_min + i * h;
  
  return grid;
}

std::vector<std::vector<double>> Romberg_table(const std::vector<double>& f_vals,
                                               double h_base,
                                               int n_levels) {
  int n = f_vals.size();
  std::vector<std::vector<double>> R(n_levels);
  
  for (int k = 0; k < n_levels; ++k)
    R[k].resize(k + 1, 0.0);
  
  for (int k = 0; k < n_levels; ++k) {
    int stride = 1 << (n_levels - 1 - k);
    double sum = 0.0;
    for (int i = 0; i < n; i += stride) {
      double w = 1.0;
      if (i == 0 || i == n - 1) w = 0.5;
      sum += w * f_vals[i];
    }
    R[k][0] = sum * h_base * stride;
  }
  
  for (int m = 1; m < n_levels; ++m) {
    for (int k = m; k < n_levels; ++k) {
      R[k][m] = R[k][m - 1] + (R[k][m - 1] - R[k - 1][m - 1]) / (std::pow(4.0, m) - 1.0);
    }
  }
  
  return R;
}

// [[Rcpp::export]]
double KL_Romberg(const std::vector<double>& x_grid,
                  const std::vector<double>& f_target,
                  const std::vector<double>& f_proposal,
                  int n_levels = 5) {
  int n = f_target.size();
  double h_base = x_grid[1] - x_grid[0];
  
  std::vector<double> integrand(n);
  double eps = 1e-300;
  
  for (int i = 0; i < n; ++i) {
    double ft = f_target[i];
    double fq = f_proposal[i];
    
    if (ft <= eps) {
      integrand[i] = 0.0;
    } else if (fq <= eps) {
      integrand[i] = 0.0; 
    } else {
      integrand[i] = ft * std::log(ft / fq);
    }
  }
  
  auto R = Romberg_table(integrand, h_base, n_levels);
  return R[n_levels - 1][n_levels - 1];
}