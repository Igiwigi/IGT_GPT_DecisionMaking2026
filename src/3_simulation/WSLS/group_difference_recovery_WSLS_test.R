# ============================================================================
# GROUP DIFFERENCE RECOVERY - HIERARCHICAL WSLS (3 groups: Human, GPT2023, GPT2025)
# PART 1: ANALYSIS & SIMULATION
# ============================================================================
# This script runs the parameter recovery simulation and saves all results (to see group difference recovery!!)
# Run visualization script separately after this completes!!
# simulating new data since old data doesn't encompass group differences (this is very stupid and should have been done in the python simulation file in the first place!!!!)

library(pacman)
pacman::p_load(extraDistr, R2jags, parallel, jsonlite, data.table)
set.seed(1234)

# ---------------------------
# PATHS
# ---------------------------
root_path <- "~/IGT_LLM_DecisionMaking"
results_dir <- file.path(root_path, "src/3_simulation/WSLS/recovery/results/group_diff")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
model_file <- file.path(root_path, "models", "simpler_models", "WSLS_hier_grp_compare.txt")

payoff_file <- file.path(root_path, "utils", "payoff_scheme_3.csv")
payoff_df <- fread(payoff_file)

# ---------------------------
# SIMULATION PARAMETERS
# ---------------------------
n_simulations <- 30
n_subjects_per_group <- 30
n_trials <- 100
group_names <- c("human", "gpt2023", "gpt2025")

# Define TRUE group parameters based on empirical data
true_params <- list(
  # Group 1: Humans
  human = list(
    mu = c(prob_stay_win_raw = qnorm(0.85),  #high
           prob_stay_lose_raw = qnorm(0.30), #lowish
           epsilon_raw = qnorm(0.27)),  #some exploration
    lambda = c(prob_stay_win = 2,
               prob_stay_lose = 2,
               epsilon = 2)
  ),
  # Group 2: GPT 2023 (old)
  gpt2023 = list(
    mu = c(prob_stay_win_raw = qnorm(0.65), #lower
           prob_stay_lose_raw = qnorm(0.50), #kinda high
           epsilon_raw = qnorm(0.19)), #some exploration (slightly less)
    lambda = c(prob_stay_win = 2,
               prob_stay_lose = 2,
               epsilon = 3)
  ),
  # Group 3: GPT 2025 (new)
  gpt2025 = list(
    mu = c(prob_stay_win_raw = qnorm(0.92), #highest
           prob_stay_lose_raw = qnorm(0.25), #low
           epsilon_raw = qnorm(0.05)), #very little exploration
    lambda = c(prob_stay_win = 4,
               prob_stay_lose = 4,
               epsilon = 10)
  )
)

# Print true parameters
cat("=== TRUE GROUP PARAMETERS ===\n")
for (group_name in names(true_params)) {
  cat(sprintf("\n%s:\n", toupper(group_name)))
  mu <- true_params[[group_name]]$mu
  lambda <- true_params[[group_name]]$lambda
  cat(sprintf("  mu: prob_stay_win=%.3f (p=%.3f), prob_stay_lose=%.3f (p=%.3f), epsilon=%.3f (p=%.3f)\n",
              mu[1], pnorm(mu[1]), mu[2], pnorm(mu[2]), mu[3], pnorm(mu[3])))
  cat(sprintf("  lambda: %.1f, %.1f, %.1f\n", lambda[1], lambda[2], lambda[3]))
}

# ---------------------------
# Helper functions
# ---------------------------
phi <- function(x) pnorm(x)
translate_payoff <- function(payoff_df, scale=TRUE){
  for(deck in c('A','B','C','D')){
    payoff_df[[paste0(deck,'_outcome')]] <- payoff_df[[paste0('win_',deck)]] - payoff_df[[paste0('loss_',deck)]]
  }
  payoff_array <- as.matrix(payoff_df[, .(A_outcome,B_outcome,C_outcome,D_outcome)])
  if(scale) payoff_array <- payoff_array/100
  return(payoff_array)
}

simulate_subject <- function(payoff_array, n_trials, prob_stay_win, prob_stay_lose, epsilon){
  x <- rep(NA, n_trials); X <- rep(NA, n_trials)
  deck_counts <- rep(1, 4)
  x[1] <- sample(1:4, 1)
  X[1] <- as.numeric(payoff_array[deck_counts[x[1]], x[1]] >= 0)
  deck_counts[x[1]] <- deck_counts[x[1]] + 1
  for(t in 2:n_trials){
    available_decks <- which(deck_counts <= nrow(payoff_array))
    if(length(available_decks) == 0) break
    if(runif(1) < epsilon){
      x[t] <- sample(available_decks, 1)
    } else {
      last_choice <- x[t-1]; last_outcome <- X[t-1]
      stay_prob <- if(last_outcome==1) prob_stay_win else prob_stay_lose
      if(runif(1) < stay_prob && last_choice %in% available_decks){
        x[t] <- last_choice
      } else {
        switch_options <- setdiff(available_decks, last_choice)
        if(length(switch_options)==0) switch_options <- available_decks
        x[t] <- sample(switch_options, 1)
      }
    }
    idx <- deck_counts[x[t]]
    X[t] <- as.numeric(payoff_array[idx, x[t]] >= 0)
    deck_counts[x[t]] <- deck_counts[x[t]] + 1
  }
  valid <- !is.na(X)
  list(x=x[valid], X=X[valid], n_trials=sum(valid))
}

payoff_array <- translate_payoff(payoff_df)

# ---------------------------
# Storage
# ---------------------------
results <- data.frame()
contrasts <- data.frame()
hyperparams <- data.frame()
samples_list <- list()

# ---------------------------
# Simulation + Fit Loop
# ---------------------------
start_time <- Sys.time()
cat("Simulation started:", format(start_time), "\n")

for(sim in 1:n_simulations){
  cat(sprintf("\n=== SIMULATION %d/%d ===\n", sim, n_simulations))
  all_x <- list(); all_X <- list(); all_group <- c()
  
  for(g in 1:length(group_names)){
    group_name <- group_names[g]; mu_g <- true_params[[group_name]]$mu; lambda_g <- true_params[[group_name]]$lambda
    hyperparams <- rbind(hyperparams,data.frame(
      sim_id=sim, group=group_name,
      mu_prob_stay_win=phi(mu_g[1]),
      mu_prob_stay_lose=phi(mu_g[2]),
      mu_epsilon=phi(mu_g[3]),
      lambda_prob_stay_win=lambda_g[1],
      lambda_prob_stay_lose=lambda_g[2],
      lambda_epsilon=lambda_g[3]
    ))
    
    for(s in 1:n_subjects_per_group){
      prob_stay_win_s <- phi(rnorm(1, mu_g[1], 1/sqrt(lambda_g[1])))
      prob_stay_lose_s <- phi(rnorm(1, mu_g[2], 1/sqrt(lambda_g[2])))
      epsilon_s <- phi(rnorm(1, mu_g[3], 1/sqrt(lambda_g[3])))
      sim_s <- simulate_subject(payoff_array, n_trials, prob_stay_win_s, prob_stay_lose_s, epsilon_s)
      all_x[[length(all_x)+1]] <- sim_s$x
      all_X[[length(all_X)+1]] <- sim_s$X
      all_group <- c(all_group,g)
    }
  }
  
  max_trials <- max(sapply(all_X,length))
  x_mat <- t(sapply(all_x, function(v){c(v, rep(NA,max_trials-length(v)))}))
  X_mat <- t(sapply(all_X, function(v){c(v, rep(NA,max_trials-length(v)))}))
  
  jags_data <- list(
    x = x_mat, X = X_mat,
    ntrials = apply(!is.na(X_mat),1,sum),
    nsubs = length(all_x), group = all_group,
    n_groups = 3
  )
  
  jags_params <- c("mu_prob_stay_win_raw","mu_prob_stay_lose_raw","mu_epsilon_raw",
                   "lambda_prob_stay_win","lambda_prob_stay_lose","lambda_epsilon")
  
  cat("Running JAGS...\n")
  sim_start <- Sys.time()
  samples <- jags.parallel(
    data=jags_data, inits=NULL, parameters.to.save=jags_params,
    model.file=model_file,
    n.chains=3, n.iter=5000, n.burnin=1000, n.thin=1, n.cluster=3,
    jags.seed=1234
  )
  cat(sprintf("Completed in %.2f minutes\n", as.numeric(difftime(Sys.time(), sim_start, units="mins"))))
  
  samples_list[[sim]] <- samples$BUGSoutput$sims.list
  summary_mat <- samples$BUGSoutput$summary
  
  # ---------------------------
  # Extract group-level parameters & convergence
  # ---------------------------
  for(g in 1:length(group_names)){
    group_name <- group_names[g]
    true_mu <- true_params[[group_name]]$mu
    true_lambda <- true_params[[group_name]]$lambda
    param_rows_group <- grep(paste0("\\[",g,"\\]$"), rownames(summary_mat))
    
    # Posterior medians
    rec_mu_prob_stay_win <- median(samples$BUGSoutput$sims.list$mu_prob_stay_win_raw[,g])
    rec_mu_prob_stay_lose <- median(samples$BUGSoutput$sims.list$mu_prob_stay_lose_raw[,g])
    rec_mu_epsilon <- median(samples$BUGSoutput$sims.list$mu_epsilon_raw[,g])
    
    results <- rbind(results,data.frame(
      sim_id=sim, group=group_name,
      true_mu_prob_stay_win=true_mu[1],
      true_mu_prob_stay_lose=true_mu[2],
      true_mu_epsilon=true_mu[3],
      true_lambda_prob_stay_win=true_lambda[1],
      true_lambda_prob_stay_lose=true_lambda[2],
      true_lambda_epsilon=true_lambda[3],
      rec_mu_prob_stay_win=rec_mu_prob_stay_win,
      rec_mu_prob_stay_lose=rec_mu_prob_stay_lose,
      rec_mu_epsilon=rec_mu_epsilon,
      rec_lambda_prob_stay_win=median(samples$BUGSoutput$sims.list$lambda_prob_stay_win[,g]),
      rec_lambda_prob_stay_lose=median(samples$BUGSoutput$sims.list$lambda_prob_stay_lose[,g]),
      rec_lambda_epsilon=median(samples$BUGSoutput$sims.list$lambda_epsilon[,g]),
      max_rhat=max(summary_mat[param_rows_group,"Rhat"],na.rm=TRUE),
      min_ess=min(summary_mat[param_rows_group,"n.eff"],na.rm=TRUE)
    ))
  }
  
  # ---------------------------
  # Extract group contrasts
  # ---------------------------
  param_names <- c("prob_stay_win","prob_stay_lose","epsilon")
  for(param in param_names){
    param_name <- paste0("mu_",param,"_raw")
    rec_vals <- sapply(1:length(group_names),function(g) median(samples$BUGSoutput$sims.list[[param_name]][,g]))
    true_vals <- sapply(group_names, function(gn) true_params[[gn]]$mu[paste0(param,"_raw")])
    contrasts <- rbind(contrasts,data.frame(
      sim_id=sim, parameter=param,
      true_diff_gpt2023_human=true_vals[2]-true_vals[1],
      true_diff_gpt2025_human=true_vals[3]-true_vals[1],
      true_diff_gpt2025_gpt2023=true_vals[3]-true_vals[2],
      rec_diff_gpt2023_human=rec_vals[2]-rec_vals[1],
      rec_diff_gpt2025_human=rec_vals[3]-rec_vals[1],
      rec_diff_gpt2025_gpt2023=rec_vals[3]-rec_vals[2]
    ))
  }
  
  # ---------------------------
  # Probability-scale recovery table
  # ---------------------------
  for(g in 1:length(group_names)){
    group_name <- group_names[g]
    results[results$group==group_name, c("rec_mu_prob_stay_win_prob",
                                         "rec_mu_prob_stay_lose_prob",
                                         "rec_mu_epsilon_prob")] <-
      t(apply(results[results$group==group_name, c("rec_mu_prob_stay_win","rec_mu_prob_stay_lose","rec_mu_epsilon")],1,phi))
    
    results[results$group==group_name, c("true_mu_prob_stay_win_prob",
                                         "true_mu_prob_stay_lose_prob",
                                         "true_mu_epsilon_prob")] <-
      t(apply(results[results$group==group_name, c("true_mu_prob_stay_win","true_mu_prob_stay_lose","true_mu_epsilon")],1,phi))
  }
  
  # ---------------------------
  # Save incremental results
  # ---------------------------
  write.csv(results, file.path(results_dir,"group_params_recovery.csv"), row.names=FALSE)
  write.csv(contrasts, file.path(results_dir,"group_contrasts_recovery.csv"), row.names=FALSE)
  write.csv(hyperparams, file.path(results_dir,"group_hyperparams.csv"), row.names=FALSE)
}

# ---------------------------
# Save posteriors
# ---------------------------
saveRDS(samples_list, file.path(results_dir,"samples_list.rds"))
saveRDS(true_params, file.path(results_dir,"true_params.rds"))

# ============================================================================
# COMPUTE BIAS / RMSE / CORRELATION METRICS
# ============================================================================

# ---------------------------
# Group parameter recovery metrics
# ---------------------------
cat("\n=== GROUP PARAMETER RECOVERY METRICS ===\n")
group_metrics <- data.frame()

for(group_name in group_names){
  res_g <- results[results$group == group_name, ]
  
  for(param in param_names){
    rec <- res_g[[paste0("rec_mu_", param)]]
    true <- res_g[[paste0("true_mu_", param)]]
    
    rec_prob <- phi(rec)
    true_prob <- phi(true)
    
    bias <- mean(rec_prob - true_prob, na.rm=TRUE)
    rmse <- sqrt(mean((rec_prob - true_prob)^2, na.rm=TRUE))
    
    cat(sprintf("%s | %s: bias=%.3f, RMSE=%.3f\n", toupper(group_name), param, bias, rmse))
    
    group_metrics <- rbind(group_metrics, data.frame(
      group = group_name,
      parameter = param,
      bias = bias,
      rmse = rmse
    ))
  }
}

write.csv(group_metrics, file.path(results_dir, "group_parameter_recovery_metrics.csv"), row.names=FALSE)

# ---------------------------
# Group contrast recovery metrics
# ---------------------------
cat("\n=== GROUP CONTRAST RECOVERY METRICS ===\n")
contrast_names <- c("gpt2023_human", "gpt2025_human", "gpt2025_gpt2023")
contrast_metrics <- data.frame()

for(param in param_names){
  con_p <- contrasts[contrasts$parameter == param, ]
  
  for(contrast in contrast_names){
    true_col <- paste0("true_diff_", contrast)
    rec_col <- paste0("rec_diff_", contrast)
    
    r <- cor(con_p[[true_col]], con_p[[rec_col]], use="complete.obs")
    bias <- mean(con_p[[rec_col]] - con_p[[true_col]], na.rm=TRUE)
    rmse <- sqrt(mean((con_p[[rec_col]] - con_p[[true_col]])^2, na.rm=TRUE))
    
    cat(sprintf("%s | %s: bias=%.3f, r=%.3f, RMSE=%.3f\n", param, contrast, bias, r, rmse))
    
    contrast_metrics <- rbind(contrast_metrics, data.frame(
      parameter = param,
      contrast = contrast,
      bias = bias,
      r = r,
      rmse = rmse
    ))
  }
}

write.csv(contrast_metrics, file.path(results_dir, "group_contrast_recovery_metrics.csv"), row.names=FALSE)

cat("\n=== ALL SIMULATIONS COMPLETE ===\n")
end_time <- Sys.time()
cat(sprintf("Total time: %.2f minutes\n", as.numeric(difftime(end_time, start_time, units="mins"))))