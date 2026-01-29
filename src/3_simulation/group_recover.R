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
  # mode estimator (finds the most likely value from a distr of MCMC samples)
  x <- as.numeric(x)
  if(length(x) == 0 || all(is.na(x))) return(NA)
  d <- density(x)
  d$x[which.max(d$y)]
}

# ---------------------------
# SETUP
# ---------------------------
n_groups <- length(data$x) #100
#for testing
n_groups <- 30 #100 is overkill, maybe do 30 in the end (only a few for testing though)

# set timer
start_time <- Sys.time()
print(paste0("this is when this started running: ", start_time))

# ---------------------------
# INIT PARAM STORAGE (matching the new model)
# ---------------------------
params <- c("mu_a_rew", "mu_a_pun", "mu_K", "mu_omega_f", "mu_omega_p") #grp lvl estimates (group mean)
sigma_params <- c("sigma_a_rew", "sigma_a_pun", "sigma_K", "sigma_omega_f", "sigma_omega_p") #grp lvl SDs

true_mu <- matrix(NA, nrow=n_groups, ncol=length(params))
infer_mu <- matrix(NA, nrow=n_groups, ncol=length(params))

true_sd  <- matrix(NA, nrow=n_groups, ncol=length(sigma_params))
infer_sd <- matrix(NA, nrow=n_groups, ncol=length(sigma_params))

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
  
  sigma_a_rew <- as.numeric(data$sigma_a_rew[i])
  sigma_a_pun <- as.numeric(data$sigma_a_pun[i])
  sigma_K <- as.numeric(data$sigma_K[i])
  sigma_omega_f <- as.numeric(data$sigma_omega_f[i])
  sigma_omega_p <- as.numeric(data$sigma_omega_p[i])
  
  #prepare jags
  jags_data <- list(
    x = x,
    X = X,
    ntrials = ntrials,
    nsubs = nsubs
  )
  
  #params to monitor (explicitly listed - this is critical for JAGS to track them!)
  if (fixed_theta) {
    jags_params <- c("mu_a_rew", "mu_a_pun", "mu_K", "mu_omega_f", "mu_omega_p",
                     "sigma_a_rew", "sigma_a_pun", "sigma_K", "sigma_omega_f", "sigma_omega_p")
    model_file <- file.path(root_path, "models", "hier_ORL_fixed_theta_new_ver.txt")
  } else {
    jags_params <- c("mu_a_rew", "mu_a_pun", "mu_K", "mu_omega_f", "mu_omega_p", "mu_theta",
                     "sigma_a_rew", "sigma_a_pun", "sigma_K", "sigma_omega_f", "sigma_omega_p", "sigma_theta")
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
  
  #true and inferred group means
  true_mu[i, ] <- c(mu_a_rew, mu_a_pun, mu_K, mu_omega_f, mu_omega_p)
  infer_mu[i, ] <- sapply(params, function(p) MPD(sims[[p]])) #check if this works (it might not)
  
  #true and inferred group SDs
  true_sd[i, ] <- c(sigma_a_rew, sigma_a_pun, sigma_K, sigma_omega_f, sigma_omega_p)
  infer_sd[i, ] <- sapply(sigma_params, function(p) MPD(sims[[p]]))
}

#save results
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
  
  true_sd_a_rew = true_sd[,1],
  true_sd_a_pun = true_sd[,2],
  true_sd_K = true_sd[,3],
  true_sd_omega_f = true_sd[,4],
  true_sd_omega_p = true_sd[,5],
  
  infer_sd_a_rew = infer_sd[,1],
  infer_sd_a_pun = infer_sd[,2],
  infer_sd_K = infer_sd[,3],
  infer_sd_omega_f = infer_sd[,4],
  infer_sd_omega_p = infer_sd[,5]
)

savefile_name <- paste0("param_recovery_group_ALL_", if(fixed_theta) "fixed_theta", ".csv")
write.csv(df, file.path(root_path, "src/3_simulation/recovery/results", savefile_name), row.names=FALSE)
cat("Parameter recovery results saved to", savefile_name, "\n")

# print time run
end_time <- Sys.time()
print(end_time - start_time)

#corr & plots
cor(df$true_mu_a_rew, df$infer_mu_a_rew)
cor(df$true_mu_a_pun, df$infer_mu_a_pun)
cor(df$true_mu_K, df$infer_mu_K)
cor(df$true_mu_omega_f, df$infer_mu_omega_f)
cor(df$true_mu_omega_p, df$infer_mu_omega_p)

cor(df$true_sd_a_rew, df$infer_sd_a_rew)
cor(df$true_sd_a_pun, df$infer_sd_a_pun)
cor(df$true_sd_K, df$infer_sd_K)
cor(df$true_sd_omega_f, df$infer_sd_omega_f)
cor(df$true_sd_omega_p, df$infer_sd_omega_p)

#plot mu parameters
png(file.path(root_path, "src/3_simulation/recovery/results",
              paste0("recovery_mu_", if(fixed_theta) "fixed_theta", ".png")),
    width = 2100, height = 600, res = 150)

par(mfrow=c(1,5), mar=c(3,3,2,1))
for(p in c("a_rew", "a_pun", "K", "omega_f", "omega_p")) {
  true_vals <- df[[paste0("true_mu_", p)]]
  infer_vals <- df[[paste0("infer_mu_", p)]]
  plot(true_vals, infer_vals, main=paste("mu", p), xlab="True", ylab="Inferred")
  abline(a=0, b=1, col="red")
}
dev.off()


#plot sigma parameters
if(!is.na(sigma_a_pun)){ #only plot if sigma was estimated
  png(file.path(root_path, "src/3_simulation/recovery/results",
                paste0("recovery_sigma_", if(fixed_theta) "fixed_theta", ".png")),
      width = 2100, height = 600, res = 150)
  
  par(mfrow=c(1,5), mar=c(3,3,2,1))
  for(p in c("a_rew", "a_pun", "K", "omega_f", "omega_p")) {
    true_vals <- df[[paste0("true_sd_", p)]]
    infer_vals <- df[[paste0("infer_sd_", p)]]
    plot(true_vals, infer_vals, main=paste("sigma", p), xlab="True", ylab="Inferred")
    abline(a=0, b=1, col="red")
  }
  dev.off()
}
