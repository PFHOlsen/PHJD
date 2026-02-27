################################################################################
# Example: HEJD fit to Meixner process #########################################
################################################################################

# Source C++ code
Rcpp::sourceCpp("PHJD.cpp")
Rcpp::sourceCpp("HE_EM.cpp")

# Load packages
library(matrixdist)

################################################################################
# Model setup ##################################################################
################################################################################

# Meixner parameters
a <- 0.3977   # a > 0
b <- -1.4940  # -pi < b < pi
d <- 0.3462   # d > 0
m <- 0        # M any real numbers

# Lower limits for support of references measures
# Choose A strictly larger than their maximum
lower_up <- pi/a-b/a
lower_down <- pi/a+b/a

# Mean value of target meixner distribution X(1)
mean_target <- m + d * a * tan(b/2)

# Density function of target meixner distribution X(1)
fX1 <- function(x){
  d1 <- (2*cos(b/2)) ** (2*d) / (2*a*pi*gamma(2*d))
  d2 <- exp(b*(x-m)/a) * abs(pracma::gammaz(d + 1i * (x-m)/a))**2
  return(d1*d2)    
}

# Equidistant grid for Kullback-Leibler computation using Romberg integration
RG <- Romberg_grid(x_min = -2,    # lower endpoint of integration ranee
                   x_max = 1,     # upper endpoint of integration range
                   n_base = 500L, # Number of grid points in base level
                   n_levels = 5L  # Number of levels (Romgerg order)
                   )

# Density function for target meixner distribution X(1) evaluated in Romberg grid
f_target <- fX1(RG)

################################################################################
# Main function for fitting ####################################################
################################################################################

HEJD_fit <- function(A,      # Truncation point
                     p,      # Dimension of HE representations
                     q,      # Number of Erlang phases in density approximation
                     N_steps # Number of EM steps
                     ){
  
  # Specifications for numerical integration using "integrate" in R
  rel.tol = 10**(-12)
  abs.tol = 10**(-12)
  subdivisions = 10**6
  
  # Define reference densities for positive and negative jumps
  v_up <- function(x){2*d*ceiling(a*(x-pi/a + b/a)/(2*pi))} # t> pi/a - b/a
  v_down <- function(x){2*d*ceiling(a*(x-pi/a - b/a)/(2*pi))} # t> pi/a + b/a
  
  # Compute asymptotics variance for contribution of small jumps
  (varA_up <- 2 * integrate(f = function(t){
    log_val <- log(v_up(t)) - 3 * log(t)
    return(exp(log_val))
  }, 
  rel.tol = rel.tol, abs.tol = abs.tol, subdivisions = subdivisions,
  lower = A, upper = Inf)[[1]])
  
  varA_down <- 2 * integrate(f = function(t){
    log_val <- log(v_down(t)) - 3 * log(t)
    return(exp(log_val))
  },
  rel.tol = rel.tol, abs.tol = abs.tol, subdivisions = subdivisions,
  lower = A, upper = Inf)[[1]]
  
  # Define truncated Lévy densities  
  nA_up <- function(x){integrate(f = function(t){
    log_val <- - x * t + log(v_up(t))
    return(exp(log_val))
  },
  rel.tol = rel.tol, abs.tol = abs.tol, subdivisions = subdivisions,
  lower = lower_up, upper = A)[[1]]}
  
  nA_down <- function(x){integrate(f = function(t){
    log_val <- - x * t + log(v_down(t))
    return(exp(log_val))
  },
  rel.tol = rel.tol, abs.tol = abs.tol, subdivisions = subdivisions,
  lower = lower_down, upper = A)[[1]]}
  
  # Compute Poisson rates
  rho_up <<- integrate(f = function(t){
    v_up(t)/t
  },  
  rel.tol = rel.tol, abs.tol = abs.tol, subdivisions = subdivisions,
  lower = lower_up, upper = A)[[1]]
  
  rho_down <<- integrate(f = function(t){
    v_down(t)/t
  },
  rel.tol = rel.tol, abs.tol = abs.tol, subdivisions = subdivisions,
  lower = lower_down, upper = A)[[1]]
  
  # Define truncated and normalised Lévy densities
  nA_dens_up <- function(x){nA_up(x)/rho_up}
  nA_dens_down <- function(x){nA_down(x)/rho_down}
  
  # Helper function for EM algorithm for HE distributions
  fit_HE <- function(f, y, p){
    set.seed(1)
    # Set initial HE parameters
    alpha_vec <- rexp(p,1)
    alpha_vec <- alpha_vec / sum(alpha_vec)
    mu <- rexp(p,1)
    # Compute EM weight from density
    w <- sapply(X = y, FUN = f)
    # Iterate EM algorithm for HE distributions
    for(i in 1:N_steps){
      EM_HE(y, w, alpha_vec, mu)
      cat("\r", paste0("Iter: ", i))
    }
    # Convert to ph object in matrixdist package
    HE_fit <- matrixdist::ph(alpha = alpha_vec, S = -diag(mu))
    return(HE_fit)
  }
  
  # Find upper and lower limit for truncation of nA_dens_up and nA_dens_down
  find_T <- function(eps, f){
    uniroot(f = function(x){f(x)-eps}, 
            lower = 0, upper = 10, 
            maxiter = 1000, tol = 10**(-12))$root
  }
  
  T_max_up <-   find_T(eps = 10**(-4), nA_dens_up)
  T_max_down <- find_T(eps = 10**(-4), nA_dens_down)
  
  T_min_up <-   T_max_up   / 10**4
  T_min_down <- T_max_down / 10**4
  
  # Compute grid for discretisation of nA_dens_up and nA_dens_down
  y_up <-   seq(T_min_up,   T_max_up,   length.out = 2000)
  y_down <- seq(T_min_down, T_max_down, length.out = 2000)
  
  # Fit HE representations to nA_dens_up and nA_dens_down
  HE_up   <<- fit_HE(f = nA_dens_up,   y = y_up, p = p)
  HE_down <<- fit_HE(f = nA_dens_down, y = y_down, p = p)
  
  plot(y_up, sapply(X = y_up, FUN = nA_dens_up), 
       type = "l", xlab = "", ylab = "", lwd = 2, col = "#901a1E",
       main = "HE approximation for positive jumps")
  lines(y_up, matrixdist::dens(HE_up, y_up), col = "#901a1E", lwd = 2)
  plot(y_down, sapply(X = y_down, FUN = nA_dens_down),
       type = "l", xlab = "", ylab = "", lwd = 2, col = "#901a1E",
       main = "HE approximation for negative jumps")
  lines(y_down, matrixdist::dens(HE_down, y_down), col = "#901a1E", lwd = 2)
  
  # Means of fitted HE jump components
  mean_up   <- matrixdist::moment(HE_up,   k = 1)
  mean_down <- matrixdist::moment(HE_down, k = 1)
  
  # Mean of compound poisson component of HEJD
  mean_jumps <- rho_up * mean_up - rho_down * mean_down
  
  # Set drift and diffusion of HEJD
  drift <<- mean_target - mean_jumps
  sig2 <<- varA_up + varA_down
  
  # Compute parameters for density by fixed point equation from:
  # Asmussen (2024), "Erlangization/Canadization of Phase-Type Jump Diffusions, 
  # with Applications to Barrier Options"
  eta <- q
  par_up <- psi_fixed_point(p = p, q = q, eta = eta,
                            a = drift, sig2 = sig2,
                            lambda_up = rho_up , lambda_down = rho_down,
                            alpha_up = HE_up@pars$alpha, T_up = HE_up@pars$S,
                            alpha_down = HE_down@pars$alpha, T_down = HE_down@pars$S)
  
  par_down <- psi_fixed_point(p = p, q = q, eta = eta,
                              a = -drift, sig2 = sig2,
                              lambda_up = rho_down , lambda_down = rho_up,
                              alpha_up = HE_down@pars$alpha, T_up = HE_down@pars$S,
                              alpha_down = HE_up@pars$alpha, T_down = HE_up@pars$S)
  
  # Extract G matrices
  G_up <<- par_up$G
  G_down <<- par_down$G
  
  # Fast compuation of density for equidistant data points
  f_proposal <<- X_dens_equi(x = RG, G_up = G_up, G_down = G_down, q = q)
  
  # Compute Kullback-Leibker divergence by Romgerg integration
  KL <<- KL_Romberg(x_grid = RG,             # grid points for integration
                    f_target = f_target,     # Target (Meixner) density X(1)
                    f_proposal = f_proposal) # Proposal (HEJD) density Y(1)
                      
}

################################################################################
# Fitting and results ##########################################################
################################################################################

# Fit HEJD Y(1) to target Meixner distribution X(1)
fitted_HEJD <- HEJD_fit(A = 22, 
                        p = 3, 
                        q = 10 , 
                        N_step = 20000)

# Kullback-Leibler divergence (scaled)
round(KL*10**3,4)

# Fitted parameters (scaled)
# Drift and diffusion
round(drift*10,4)
round(sig2*10**2,4)
# Poisson drift
round(rho_up,4)
round(rho_down,4)
# Initial distributions
round(HE_up@pars$alpha,4)
round(HE_down@pars$alpha,4)
# HE rates
-diag(round((HE_up@pars$S),4))
-diag(round((HE_down@pars$S),4))

# Produce density figure
{
  RG_idx <- 1:length(RG) %% 50 == 0
  plot(RG[RG_idx], f_target[RG_idx], 
       type = "b",
       xlim = c(-2,1), ylim = c(0,3), lwd = 1, col = "#666666", lty = 1,
       main = "Densitiy for fitted HEJD versus Meixner \n (S&P 500)",
       ylab = "",
       xlab = "",
       cex.lab = 1.2)
  lines(RG, f_proposal, col = "#901a1E", lty = 1, lwd = 2)
  legend("topleft",
         legend = c("Meixner","HEJD"),
         col = c("#666666", "#901a1E"),
         bty = "n",
         lty = c(NA,1),
         pch = c(1, NA),
         lwd = c(1,2),
         border = "n",
         cex = 1,
         text.col = "black",
         horiz = FALSE ,
         inset = c(0.01))
}

################################################################################
# Additional functions #########################################################
################################################################################

# Evaluation of density for data points with arbitrary distance
# (Slower than X_dens_equi)
X_dens(x = RG, G_up = G_up, G_down = G_down, q = 10)
# Evaluaiton of c.d.f at data points x
X_cdf(x = RG, G_up = G_up, G_down = G_down, q = 10)
# Evaluation of u-quantiles
# The arguments lower / upper are end points for functional inversion by 
# bisection method
X_quantile(u = (1:99)/100, G_up = G_up, G_down = G_down, q = 10, lower = -10, upper = 10)
