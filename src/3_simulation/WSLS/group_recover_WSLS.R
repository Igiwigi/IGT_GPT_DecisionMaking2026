# ============================================================================
# GROUP PARAMETER RECOVERY - HIERARCHICAL WSLS (win-stay / lose-stay + epsilon)
# ============================================================================
library(pacman)
pacman::p_load(extraDistr, R2jags, parallel, jsonlite)

# setup
set.seed(1234)

root_path <- "~/IGT_LLM_DecisionMaking"
filename <- "simulated_group_winlose.json"
file <- file.path(root_path, "src/3_simulation/WSLS/recovery/simulated_data", filename)

# Load JSON simulated group data
data <- jsonlite::fromJSON(file)

# ---------------------------
# HELPER FUNCTION
# ---------------------------
MPD <- function(x) {
  x <- as.numeric(x)
  if(length(x) == 0 || all(is.na(x))) return(NA)
  d <- density(x)
  d$x[which.max(d$y)]
}

# ---------------------------
# SETUP
# ---------------------------
n_groups <- length(data$x)
n_groups <- 100 # for testing

start_time <- Sys.time()
print(paste0("Simulation started: ", start_time))

# ---------------------------
# INIT PARAM STORAGE
# ---------------------------
params <- c("mu_prob_stay_win_raw", "mu_prob_stay_lose_raw", "mu_epsilon_raw")
lambda_params <- c("lambda_prob_stay_win", "lambda_prob_stay_lose", "lambda_epsilon")

true_mu <- matrix(NA, nrow=n_groups, ncol=length(params))
infer_mu <- matrix(NA, nrow=n_groups, ncol=length(params))

true_lambda <- matrix(NA, nrow=n_groups, ncol=length(lambda_params))
infer_lambda <- matrix(NA, nrow=n_groups, ncol=length(lambda_params))

# output paths
results_dir <- file.path(root_path, "src/3_simulation/WSLS/recovery/results")
csv_file <- file.path(results_dir, "param_recovery_group_WSLS_incremental.csv")
plot_mu_file <- file.path(results_dir, "recovery_mu_WSLS_incremental.png")
plot_lambda_file <- file.path(results_dir, "recovery_lambda_WSLS_incremental.png")

# ---------------------------
# LOOP OVER GROUPS
# ---------------------------
for (i in 1:n_groups) {
  cat(sprintf("\n=== Processing group %d/%d ===\n", i, n_groups))
  
  # extract choices and outcomes
  x <- data$x[[i]] + 1
  X <- data$X[[i]]
  nsubs <- nrow(x)
  ntrials <- rep(ncol(x), nsubs)
  
  # extract true group parameters
  mu_prob_stay_win_raw <- as.numeric(data$mu_prob_stay_win_raw[i])
  mu_prob_stay_lose_raw <- as.numeric(data$mu_prob_stay_lose_raw[i])
  mu_epsilon_raw <- as.numeric(data$mu_epsilon_raw[i])
  
  lambda_prob_stay_win <- as.numeric(data$lambda_prob_stay_win[i])
  lambda_prob_stay_lose <- as.numeric(data$lambda_prob_stay_lose[i])
  lambda_epsilon <- as.numeric(data$lambda_epsilon[i])
  
  # prepare JAGS data
  jags_data <- list(
    x = x,
    X = X,
    ntrials = ntrials,
    nsubs = nsubs
  )
  
  # parameters to monitor
  jags_params <- c(params, lambda_params)
  model_file <- file.path(root_path, "models", "simpler_models", "WSLS_hier.txt") # WSLS JAGS model
  
  # run JAGS
  cat("Running JAGS ...\n")
  group_start <- Sys.time()
  
  samples <- jags.parallel(
    data = jags_data,
    inits = NULL,
    parameters.to.save = jags_params,
    model.file = model_file,
    n.chains = 3,
    n.iter = 3000, #can lower this (might not need to for WSLS)
    n.burnin = 1000,
    n.thin = 1,
    n.cluster = 4
  )
  
  group_end <- Sys.time()
  cat(sprintf("Group %d completed in %.2f minutes\n", i, as.numeric(difftime(group_end, group_start, units="mins"))))
  
  sims <- samples$BUGSoutput$sims.list
  
  # store true and inferred group means
  true_mu[i, ] <- c(mu_prob_stay_win_raw, mu_prob_stay_lose_raw, mu_epsilon_raw)
  infer_mu[i, ] <- sapply(params, function(p) MPD(sims[[p]]))
  
  # store true and inferred group lambdas
  true_lambda[i, ] <- c(lambda_prob_stay_win, lambda_prob_stay_lose, lambda_epsilon)
  infer_lambda[i, ] <- sapply(lambda_params, function(p) MPD(sims[[p]]))
  
  # ---------------------------
  # SAVE INCREMENTAL RESULTS
  # ---------------------------
  df <- data.frame(
    group_id = 1:n_groups,
    true_mu_prob_stay_win_raw = true_mu[,1],
    true_mu_prob_stay_lose_raw = true_mu[,2],
    true_mu_epsilon_raw = true_mu[,3],
    infer_mu_prob_stay_win_raw = infer_mu[,1],
    infer_mu_prob_stay_lose_raw = infer_mu[,2],
    infer_mu_epsilon_raw = infer_mu[,3],
    true_lambda_prob_stay_win = true_lambda[,1],
    true_lambda_prob_stay_lose = true_lambda[,2],
    true_lambda_epsilon = true_lambda[,3],
    infer_lambda_prob_stay_win = infer_lambda[,1],
    infer_lambda_prob_stay_lose = infer_lambda[,2],
    infer_lambda_epsilon = infer_lambda[,3]
  )
  
  write.csv(df, csv_file, row.names = FALSE)
  cat(sprintf("Saved incremental results to %s\n", basename(csv_file)))
  
  # ---------------------------
  # UPDATE PLOTS (only completed groups)
  # ---------------------------
  completed <- !is.na(infer_mu[,1])
  n_completed <- sum(completed)
  
  if(n_completed >= 2) {
    # plot mu parameters
    png(plot_mu_file, width = 1800, height = 600, res = 150)
    par(mfrow=c(1,3), mar=c(3,3,2,1))
    for(p in c("prob_stay_win_raw", "prob_stay_lose_raw", "epsilon_raw")) {
      true_vals <- df[[paste0("true_mu_", p)]][completed]
      infer_vals <- df[[paste0("infer_mu_", p)]][completed]
      plot(true_vals, infer_vals, main=p, xlab="True", ylab="Inferred")
      abline(a=0, b=1, col="red")
      if(n_completed >= 3) {
        r <- cor(true_vals, infer_vals, use="complete.obs")
        legend("topleft", legend=sprintf("r=%.3f", r), bty="n", cex=0.8)
      }
    }
    dev.off()
    
    # plot lambda parameters
    png(plot_lambda_file, width = 1800, height = 600, res = 150)
    par(mfrow=c(1,3), mar=c(3,3,2,1))
    for(p in c("prob_stay_win", "prob_stay_lose", "epsilon")) {
      true_vals <- df[[paste0("true_lambda_", p)]][completed]
      infer_vals <- df[[paste0("infer_lambda_", p)]][completed]
      plot(true_vals, infer_vals, main=p, xlab="True", ylab="Inferred")
      abline(a=0, b=1, col="red")
      if(n_completed >= 3) {
        r <- cor(true_vals, infer_vals, use="complete.obs")
        legend("topleft", legend=sprintf("r=%.3f", r), bty="n", cex=0.8)
      }
    }
    dev.off()
    
    cat(sprintf("Updated plots with %d groups\n", n_completed))
  }
  
  # print progress
  elapsed <- as.numeric(difftime(Sys.time(), start_time, units="mins"))
  avg_time_per_group <- elapsed / i
  remaining <- n_groups - i
  est_remaining <- avg_time_per_group * remaining
  cat(sprintf("Elapsed: %.1f min | Avg: %.1f min/group | Est. remaining: %.1f min\n",
              elapsed, avg_time_per_group, est_remaining))
}

# ---------------------------
# FINAL SUMMARY
# ---------------------------
end_time <- Sys.time()
total_time <- difftime(end_time, start_time, units="mins")

cat("\n=== FINAL RESULTS ===\n")
cat(sprintf("Total runtime: %.2f minutes (%.2f hours)\n", total_time, total_time/60))
cat(sprintf("Results saved to: %s\n", csv_file))

# Correlations
completed <- !is.na(infer_mu[,1])
df_complete <- df[completed,]

cat("\nMu parameter correlations:\n")
for(p in c("prob_stay_win_raw","prob_stay_lose_raw","epsilon_raw")) {
  r <- cor(df_complete[[paste0("true_mu_", p)]], df_complete[[paste0("infer_mu_", p)]])
  cat(sprintf("  %s: r = %.3f\n", p, r))
}

cat("\nLambda parameter correlations:\n")
for(p in c("prob_stay_win","prob_stay_lose","epsilon")) {
  r <- cor(df_complete[[paste0("true_lambda_", p)]], df_complete[[paste0("infer_lambda_", p)]])
  cat(sprintf("  %s: r = %.3f\n", p, r))
}

cat("\nDone!\n")
