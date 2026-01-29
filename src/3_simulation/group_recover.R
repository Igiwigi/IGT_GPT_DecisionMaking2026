# ============================================================================
# GROUP PARAMETER RECOVERY - HIERARCHICAL ORL - incremental, non parallelized version
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

# Define output paths
results_dir <- file.path(root_path, "src/3_simulation/recovery/results")
csv_file <- file.path(results_dir, paste0("param_recovery_group_ALL_", if(fixed_theta) "fixed_theta", "_incremental.csv"))
plot_mu_file <- file.path(results_dir, paste0("recovery_mu_", if(fixed_theta) "fixed_theta", "_incremental.png"))
plot_sigma_file <- file.path(results_dir, paste0("recovery_sigma_", if(fixed_theta) "fixed_theta", "_incremental.png"))

# ---------------------------
# LOOP OVER GROUPS
# ---------------------------
print(paste0("This is how many groups there are: ", n_groups))
for (i in 1:n_groups) {
  cat(sprintf("\n=== Processing group %d/%d ===\n", i, n_groups))
  
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
  
  #params to monitor
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
  group_start <- Sys.time()
  
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
  
  group_end <- Sys.time()
  cat(sprintf("Group %d completed in %.2f minutes\n", i, as.numeric(difftime(group_end, group_start, units="mins"))))
  
  sims <- samples$BUGSoutput$sims.list
  
  #true and inferred group means
  true_mu[i, ] <- c(mu_a_rew, mu_a_pun, mu_K, mu_omega_f, mu_omega_p)
  infer_mu[i, ] <- sapply(params, function(p) MPD(sims[[p]]))
  
  #true and inferred group SDs
  true_sd[i, ] <- c(sigma_a_rew, sigma_a_pun, sigma_K, sigma_omega_f, sigma_omega_p)
  infer_sd[i, ] <- sapply(sigma_params, function(p) MPD(sims[[p]]))
  
  # ---------------------------
  # SAVE INCREMENTAL RESULTS
  # ---------------------------
  df <- data.frame(
    group_id = 1:n_groups,
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
  
  write.csv(df, csv_file, row.names=FALSE)
  cat(sprintf("Saved results to %s\n", basename(csv_file)))
  
  # ---------------------------
  # UPDATE PLOTS (only with completed groups)
  # ---------------------------
  completed <- !is.na(infer_mu[,1])
  n_completed <- sum(completed)
  
  if(n_completed >= 2) {  # need at least 2 points to plot
    
    # Plot mu parameters
    png(plot_mu_file, width = 2100, height = 600, res = 150)
    par(mfrow=c(1,5), mar=c(3,3,2,1))
    
    for(p in c("a_rew", "a_pun", "K", "omega_f", "omega_p")) {
      true_vals <- df[[paste0("true_mu_", p)]][completed]
      infer_vals <- df[[paste0("infer_mu_", p)]][completed]
      
      plot(true_vals, infer_vals, 
           main=paste("mu", p, sprintf("(n=%d)", n_completed)), 
           xlab="True", ylab="Inferred")
      abline(a=0, b=1, col="red")
      
      # Add correlation if we have enough data
      if(n_completed >= 3) {
        r <- cor(true_vals, infer_vals, use="complete.obs")
        legend("topleft", legend=sprintf("r = %.3f", r), bty="n", cex=0.8)
      }
    }
    dev.off()
    
    # Plot sigma parameters
    png(plot_sigma_file, width = 2100, height = 600, res = 150)
    par(mfrow=c(1,5), mar=c(3,3,2,1))
    
    for(p in c("a_rew", "a_pun", "K", "omega_f", "omega_p")) {
      true_vals <- df[[paste0("true_sd_", p)]][completed]
      infer_vals <- df[[paste0("infer_sd_", p)]][completed]
      
      plot(true_vals, infer_vals, 
           main=paste("sigma", p, sprintf("(n=%d)", n_completed)), 
           xlab="True", ylab="Inferred")
      abline(a=0, b=1, col="red")
      
      # Add correlation if we have enough data
      if(n_completed >= 3) {
        r <- cor(true_vals, infer_vals, use="complete.obs")
        legend("topleft", legend=sprintf("r = %.3f", r), bty="n", cex=0.8)
      }
    }
    dev.off()
    
    cat(sprintf("Updated plots with %d groups\n", n_completed))
  }
  
  # Print estimated time remaining
  elapsed <- as.numeric(difftime(Sys.time(), start_time, units="mins"))
  avg_time_per_group <- elapsed / i
  remaining_groups <- n_groups - i
  estimated_remaining <- avg_time_per_group * remaining_groups
  
  cat(sprintf("Elapsed: %.1f min | Avg: %.1f min/group | Est. remaining: %.1f min\n", 
              elapsed, avg_time_per_group, estimated_remaining))
}

# ---------------------------
# FINAL SUMMARY
# ---------------------------
end_time <- Sys.time()
total_time <- difftime(end_time, start_time, units="mins")

cat("\n=== FINAL RESULTS ===\n")
cat(sprintf("Total runtime: %.2f minutes (%.2f hours)\n", total_time, total_time/60))
cat(sprintf("Results saved to: %s\n", csv_file))

# Print correlations
completed <- !is.na(infer_mu[,1])
df_complete <- df[completed,]

cat("\nMu parameter correlations:\n")
cat(sprintf("  mu_a_rew:   r = %.3f\n", cor(df_complete$true_mu_a_rew, df_complete$infer_mu_a_rew)))
cat(sprintf("  mu_a_pun:   r = %.3f\n", cor(df_complete$true_mu_a_pun, df_complete$infer_mu_a_pun)))
cat(sprintf("  mu_K:       r = %.3f\n", cor(df_complete$true_mu_K, df_complete$infer_mu_K)))
cat(sprintf("  mu_omega_f: r = %.3f\n", cor(df_complete$true_mu_omega_f, df_complete$infer_mu_omega_f)))
cat(sprintf("  mu_omega_p: r = %.3f\n", cor(df_complete$true_mu_omega_p, df_complete$infer_mu_omega_p)))

cat("\nSigma parameter correlations:\n")
cat(sprintf("  sigma_a_rew:   r = %.3f\n", cor(df_complete$true_sd_a_rew, df_complete$infer_sd_a_rew)))
cat(sprintf("  sigma_a_pun:   r = %.3f\n", cor(df_complete$true_sd_a_pun, df_complete$infer_sd_a_pun)))
cat(sprintf("  sigma_K:       r = %.3f\n", cor(df_complete$true_sd_K, df_complete$infer_sd_K)))
cat(sprintf("  sigma_omega_f: r = %.3f\n", cor(df_complete$true_sd_omega_f, df_complete$infer_sd_omega_f)))
cat(sprintf("  sigma_omega_p: r = %.3f\n", cor(df_complete$true_sd_omega_p, df_complete$infer_sd_omega_p)))

cat("\nDone!\n")