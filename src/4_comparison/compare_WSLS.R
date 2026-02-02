# ============================================================================
# GROUP COMPARE AFTER VETTING MODEL
# ============================================================================
#WSLS script to compare the found parameters between the real groups
#Is this just the same thing as the recovery script but fitted to real data? (adapt it to real data)
# a bit of a mess (still need to finish)

library(pacman)
pacman::p_load(R2jags, parallel)
set.seed(1234)

probit_to_prob <- function(x) pnorm(x)

# ---------------------------
# PATHS & LOADING
# ---------------------------
root_path <- "~/IGT_LLM_DecisionMaking"
results_dir <- file.path(root_path, "src/4_comparison/results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
model_file <- file.path(root_path, "models", "simpler_models", "WSLS_hier_grp_compare.txt")

# read in data for all groups
old_gpt_data <- read.csv(file.path(root_path, "data", "final_data", "clean_gpt_OLD_2023.csv"))
hc_data <- read.csv(file.path(root_path, "data", "final_data", "clean_ahn_hc.csv"))
new_gpt_data <- read.csv(file.path(root_path, "data", "final_data", "clean_gpt_NEW_2025.csv"))

#don't need probabilities since we can't get them for the other model anyway
old_gpt_data <- old_gpt_data %>%
  select(-probabilities)

# ---------------------------
# SIMULATION PARAMETERS (get these right, 3000 rows in each, 30 subj id)
# ---------------------------
n_simulations <- 30 #use only 30 participants (old GPT only has 30 participants)
n_groups <- 3 #unsure of what this var was currently in the model
n_trials <- 100
group_names <- c("ahn_hc", "gpt_OLD_2023", "gpt_NEW_2025")
groups <- c(hc_data, old_gpt_data, new_gpt_data)

# ---------------------------
# Storage
# ---------------------------
results <- data.frame()
contrasts <- data.frame()
hyperparams <- data.frame()
samples_list <- list()

# ---------------------------
# Parameter Recovery Loop
# ---------------------------
start_time <- Sys.time()
cat("\n=== STARTING PARAMETER RECOVERY ===\n")
cat("Total iterations to process:", n_simulations, "\n")
cat("Analysis started:", format(start_time), "\n")

for(sim in 1:n_simulations){
  cat(sprintf("\n=== ITERATION %d/%d ===\n", sim, n_simulations))
  
  # Extract data for this iteration
  x_mat <- simulated_data$x[[sim]]
  X_mat <- simulated_data$X[[sim]]
  group_labels <- simulated_data$group[[sim]]
  
  # Convert Python 0-indexing to R 1-indexing for deck choices
  x_mat <- x_mat + 1
  
  # Create group index (1=human, 2=gpt2023, 3=gpt2025)
  group_idx <- sapply(group_labels, function(g) which(group_names == g))
  
  # Store hyperparameters for this iteration
  for(g in 1:length(group_names)){
    group_name <- group_names[g]
    mu_g <- true_params[[group_name]]$mu
    lambda_g <- true_params[[group_name]]$lambda
    
    hyperparams <- rbind(hyperparams, data.frame(
      sim_id = sim,
      group = group_name,
      mu_prob_stay_win = probit_to_prob(mu_g[1]),
      mu_prob_stay_lose = probit_to_prob(mu_g[2]),
      mu_epsilon = probit_to_prob(mu_g[3]),
      lambda_prob_stay_win = lambda_g[1],
      lambda_prob_stay_lose = lambda_g[2],
      lambda_epsilon = lambda_g[3]
    ))
  }
  
  # Prepare data for JAGS
  # Convert outcomes to binary (win=1, lose=0)
  X_binary <- ifelse(X_mat >= 0, 1, 0)
  
  jags_data <- list(
    x = x_mat,
    X = X_binary,
    ntrials = apply(!is.na(x_mat), 1, sum),
    nsubs = nrow(x_mat),
    group = group_idx,
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
  
  # Estimate time remaining
  elapsed_time <- as.numeric(difftime(Sys.time(), start_time, units="mins"))
  avg_time_per_sim <- elapsed_time / sim
  remaining_sims <- n_simulations - sim
  estimated_remaining <- avg_time_per_sim * remaining_sims
  estimated_total <- avg_time_per_sim * n_simulations
  
  cat(sprintf("Progress: %d/%d (%.1f%%) | Elapsed: %.1f min | Est. remaining: %.1f min | Est. total: %.1f min\n",
              sim, n_simulations, (sim/n_simulations)*100, 
              elapsed_time, estimated_remaining, estimated_total))
  
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
    
    #what to keep in results (true, recovered, rhat, etc)
    #kinda messy naming!!! (confusing which is probit and which is prob)
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
      true_mu_prob_stay_win_prob = probit_to_prob(true_mu[1]), #taking probability, can be omitted maybe
      true_mu_prob_stay_lose_prob = probit_to_prob(true_mu[2]),
      true_mu_epsilon_prob = probit_to_prob(true_mu[3]),
      rec_mu_prob_stay_win_prob = probit_to_prob(rec_mu_prob_stay_win),
      rec_mu_prob_stay_lose_prob = probit_to_prob(rec_mu_prob_stay_lose),
      rec_mu_epsilon_prob = probit_to_prob(rec_mu_epsilon),
      min_ess=min(summary_mat[param_rows_group,"n.eff"],na.rm=TRUE),
      max_rhat=max(summary_mat[param_rows_group,"Rhat"],na.rm=TRUE),
      rhat_prob_stay_win = summary_mat[paste0("mu_prob_stay_win_raw[",g,"]"), "Rhat"],
      rhat_prob_stay_lose = summary_mat[paste0("mu_prob_stay_lose_raw[",g,"]"), "Rhat"],
      rhat_epsilon = summary_mat[paste0("mu_epsilon_raw[",g,"]"), "Rhat"],
      rhat_lambda_prob_stay_win = summary_mat[paste0("lambda_prob_stay_win[",g,"]"), "Rhat"],
      rhat_lambda_prob_stay_lose = summary_mat[paste0("lambda_prob_stay_lose[",g,"]"), "Rhat"],
      rhat_lambda_epsilon = summary_mat[paste0("lambda_epsilon[",g,"]"), "Rhat"]
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
    
    rec_prob <- probit_to_prob(rec)
    true_prob <- probit_to_prob(true)
    
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
    
    #r <- cor(con_p[[true_col]], con_p[[rec_col]], use="complete.obs") #the sd is 0, not informative
    bias <- mean(con_p[[rec_col]] - con_p[[true_col]], na.rm=TRUE)
    rmse <- sqrt(mean((con_p[[rec_col]] - con_p[[true_col]])^2, na.rm=TRUE))
    
    cat(sprintf("%s | %s: bias=%.3f, RMSE=%.3f\n", param, contrast, bias, rmse))
    
    contrast_metrics <- rbind(contrast_metrics, data.frame(
      parameter = param,
      contrast = contrast,
      bias = bias,
      #r = r, #the SD is 0
      rmse = rmse
    ))
  }
}

write.csv(contrast_metrics, file.path(results_dir, "group_contrast_recovery_metrics.csv"), row.names=FALSE)

cat("\n=== ALL SIMULATIONS COMPLETE ===\n")
end_time <- Sys.time()
cat(sprintf("Total time: %.2f minutes\n", as.numeric(difftime(end_time, start_time, units="mins"))))