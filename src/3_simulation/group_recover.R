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

# ---------------------------
# HELPER FUNCTION
# ---------------------------
MPD <- function(x) {
  # robust mode estimator
  x <- as.numeric(x)
  if(length(x) == 0 || all(is.na(x))) return(NA)
  d <- density(x)
  d$x[which.max(d$y)]
}

# ---------------------------
# SETUP
# ---------------------------
n_groups <- length(data$x)
#for testing
#n_groups <- 1
n_groups <- 30 #100 is overkill

# set timer
start_time <- Sys.time()
print(start_time, "this is when this started running!") #too lazy to fix the error regarding the timezone

# ---------------------------
# INIT PARAM STORAGE (matching the new model)
# ---------------------------
params <- c("mu_a_rew", "mu_a_pun", "mu_K", "mu_omega_f", "mu_omega_p") #grp lvl estimates (group mean)

true_mu <- matrix(NA, nrow=n_groups, ncol=length(params))
infer_mu <- matrix(NA, nrow=n_groups, ncol=length(params))

# SDs not in current JAGS output, set as NA placeholders
true_sd  <- matrix(NA, nrow=n_groups, ncol=length(params))
infer_sd <- matrix(NA, nrow=n_groups, ncol=length(params))

# ---------------------------
# LOOP OVER GROUPS
# ---------------------------
print(paste0("This is how many groups there are: ", n_groups))
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
  
  #prepare jags
  jags_data <- list(
    x = x,
    X = X,
    ntrials = ntrials,
    nsubs = nsubs
  )
  
  #params to monitor
  if (fixed_theta) {
    jags_params <- params  # only mu_* exist in JAGS output
    model_file <- file.path(root_path, "models", "hier_ORL_fixed_theta.txt")
  } else {
    jags_params <- c(params, "mu_theta", "sigma_theta") # sigma_* not currently in output
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
  cat("Available parameters in JAGS output:\n")
  print(names(sims))
  
  #true and inferred params
  true_mu[i, ] <- c(mu_a_rew, mu_a_pun, mu_K, mu_omega_f, mu_omega_p)
  infer_mu[i, ] <- sapply(params, function(p) MPD(sims[[p]]))
  
  #store true SDs as NA placeholders (theyre not monitored in the model..?)
  true_sd[i, ] <- NA
  infer_sd[i, ] <- NA
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
  infer_mu_omega_p = infer_mu[,5]
)

savefile_name <- paste0("param_recovery_group_ALL_", if(fixed_theta) "fixed_theta", ".csv")
write.csv(df, file.path(root_path, "src/3_simulation/recovery/results", savefile_name), row.names=FALSE)
cat("Parameter recovery results saved to", savefile_name, "\n")

# print time run
end_time <- Sys.time()
print(end_time - start_time)

#print correlations between the values
cor(df$true_mu_a_rew, df$infer_mu_a_rew)
cor(df$true_mu_a_pun, df$infer_mu_a_pun)
cor(df$true_mu_K, df$infer_mu_K)
cor(df$true_mu_omega_f, df$infer_mu_omega_f)
cor(df$true_mu_omega_p, df$infer_mu_omega_p)

#plot the vals
par(mfrow=c(2,3))
for(p in c("a_rew", "a_pun", "K", "omega_f", "omega_p")) {
  true_vals <- df[[paste0("true_mu_", p)]]
  infer_vals <- df[[paste0("infer_mu_", p)]]
  plot(true_vals, infer_vals, main=p, xlab="True", ylab="Inferred")
  abline(a=0, b=1, col="red")
}


