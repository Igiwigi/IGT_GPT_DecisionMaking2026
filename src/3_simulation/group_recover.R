# ============================================================================
# GROUP PARAMETER RECOVERY - HIERARCHICAL ORL, seeing if we can recover params from the (new) model itself!!
# ============================================================================
library(pacman)
pacman::p_load(extraDistr, R2jags, parallel, jsonlite)

#setup
set.seed(1234)
fixed_theta <- TRUE

root_path <- "~/IGT_LLM_DecisionMaking"
filename <- paste0("simulated_group_data", if(fixed_theta) "_fixed_theta", ".json")
file <- file.path(root_path, "src/3_simulation/recovery/simulated_data", filename)

# Load JSON simulated group data
data <- jsonlite::fromJSON(file)

MPD <- function(x) { density(x)$x[which.max(density(x)$y)] }

n_groups <- length(data$x)

# set timer
start_time = Sys.time()
print(start_time, "this is when this started running!") #too lazy to fix the error regarding the timezone

# ---------------------------
# INIT PARAM STORAGE (matching the new model)
# ---------------------------
params <- c("mu_a_rew", "mu_a_pun", "mu_K", "mu_omega_f", "mu_omega_p")
lambda_params <- c("lambda_a_rew", "lambda_a_pun", "lambda_K", "lambda_omega_f", "lambda_omega_p")

true_mu <- matrix(NA, nrow=n_groups, ncol=length(params))
infer_mu <- matrix(NA, nrow=n_groups, ncol=length(params))

true_lambda <- matrix(NA, nrow=n_groups, ncol=length(lambda_params))
infer_lambda <- matrix(NA, nrow=n_groups, ncol=length(lambda_params))

# ---------------------------
# LOOP OVER GROUPS
# ---------------------------
for (i in 1:n_groups) {
  cat("Processing group", i, "\n")
  
  #extract choices and outcomes
  x <- data$x[[i]] + 1  # convert 0-3 to 1-4 for R
  X <- data$X[[i]]
  nsubs <- nrow(x)
  ntrials <- rep(ncol(x), nsubs)
  
  #extract true group parameters (make sure numeric)
  mu_a_rew <- as.numeric(data$mu_a_rew[i])
  mu_a_pun <- as.numeric(data$mu_a_pun[i])
  mu_K <- as.numeric(data$mu_K[i])
  mu_omega_f <- as.numeric(data$mu_omega_f[i])
  mu_omega_p  <- as.numeric(data$mu_omega_p[i])
  
  sigma_a_rew <- as.numeric(data$sigma_a_rew[i])
  sigma_a_pun <- as.numeric(data$sigma_a_pun[i])
  sigma_K     <- as.numeric(data$sigma_K[i])
  sigma_omega_f  <- as.numeric(data$sigma_omega_f[i])
  sigma_omega_p <- as.numeric(data$sigma_omega_p[i])
  
  #prepare jags
  jags_data <- list(
    x = x,
    X = X,
    ntrials = ntrials,
    nsubs = nsubs
  )
  
  #params to monitor
  if (fixed_theta) {
    jags_params <- c(params, lambda_params)
    model_file <- file.path(root_path, "models", "hier_ORL_fixed_theta.txt")
  } else {
    jags_params <- c(params, "mu_theta", "lambda_theta", lambda_params)
    model_file <- file.path(root_path, "models", "hier_ORL.txt")
  }
  
  #run jags
  cat("Running JAGS ...\n")
  samples <- jags.parallel(
    data = jags_data,
    inits = NULL,
    parameters.to.save = jags_params,
    model.file = model_file,
    n.chains = 3,
    n.iter = 3000,
    n.burnin = 1000,
    n.thin = 1,
    n.cluster = 4
  )
  
  sims <- samples$BUGSoutput$sims.list
  
  #true and inferred params
  true_mu[i, ] <- c(mu_a_rew, mu_a_pun, mu_K, mu_omega_f, mu_omega_p)
  infer_mu[i, ] <- sapply(params, function(p) MPD(sims[[p]]))
  
  true_lambda[i, ] <- 1 / c(sigma_a_rew, sigma_a_pun, sigma_K, sigma_omega_f, sigma_omega_p)
  infer_lambda[i, ] <- sapply(lambda_params, function(p) MPD(sims[[p]]))
  
}

# ---------------------------
# Save results
# ---------------------------
df <- data.frame(
  true_mu_a_rew = true_mu[,1],
  true_mu_a_pun = true_mu[,2],
  true_mu_K = true_mu[,3],
  true_mu_omega_f = true_mu[,4],
  true_mu_omega_p = true_mu[,5],
  
  infer_mu_a_rew = infer_mu[,1],
  infer_mu_a_pun = infer_mu[,2],
  infer_mu_K = infer_mu[,3],
  infer_mu_omega_f = infer_mu[,4],
  infer_mu_omega_p = infer_mu[,5],
  
  true_lambda_a_rew = true_lambda[,1],
  true_lambda_a_pun = true_lambda[,2],
  true_lambda_K = true_lambda[,3],
  true_lambda_omega_f = true_lambda[,4],
  true_lambda_omega_p = true_lambda[,5],
  
  infer_lambda_a_rew = infer_lambda[,1],
  infer_lambda_a_pun = infer_lambda[,2],
  infer_lambda_K = infer_lambda[,3],
  infer_lambda_omega_f = infer_lambda[,4],
  infer_lambda_omega_p = infer_lambda[,5]
)

savefile_name <- paste0("param_recovery_group_ALL_", if(fixed_theta) "fixed_theta", ".csv")
write.csv(df, file.path(root_path, "src/3_simulation/recovery/results", savefile_name), row.names=FALSE)
cat("Parameter recovery results saved to", savefile_name, "\n")

# print time run
end_time = Sys.time()
print(end_time - start_time)