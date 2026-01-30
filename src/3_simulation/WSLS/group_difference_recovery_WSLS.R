# ============================================================================
# GROUP DIFFERENCE RECOVERY - HIERARCHICAL WSLS (3 groups: Human, GPT2023, GPT2025)
# ============================================================================
# To see whether we can recover the group differences between the 3 different groups!!
# group 1 = human, 2 = gpt2023, 3 = gpt2025
# a bit of a mess again, but should work as intended (probably some double work; better to be safe than sorry)
# needs some cleanup

library(pacman)
pacman::p_load(extraDistr, R2jags, parallel, jsonlite, ggplot2, reshape2)
set.seed(1234)

# ---------------------------
# PATHS
# ---------------------------
root_path <- "~/IGT_LLM_DecisionMaking"
results_dir <- file.path(root_path, "src/3_simulation/WSLS/recovery/results/group_diff")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
model_file <- file.path(root_path, "models", "simpler_models", "WSLS_hier_grp_compare.txt") #diff model that actually keeps track of groups

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
MPD <- function(x) { #mode, switched to median for a lot of the vars
  x <- as.numeric(x)
  if(length(x) == 0 || all(is.na(x))) return(NA)
  d <- density(x)
  d$x[which.max(d$y)]
}

# ---------------------------
# SIMULATION PARAMETERS
# ---------------------------
n_simulations <- 30 #how many to do (for testing, do less)
n_subjects_per_group <- 30 #should this be slightly increased to recover param easier? (especially epsilon for new GPT)
n_trials <- 100

# Define TRUE group parameters based on empirical data!!
# probability means (mu) and probability sd (lambda)
# Group 1: Humans
true_mu_human <- c(
  prob_stay_win_raw = qnorm(0.85),    # around 0.85 in PPC (highish)
  prob_stay_lose_raw = qnorm(0.30),   # low staylose in PPC
  epsilon_raw = qnorm(0.27)           # fairly high exploration
)
true_lambda_human <- c( #keep the variability level for humans as a comparison point of sorts (fairly variable)
  prob_stay_win = 2,
  prob_stay_lose = 2,
  epsilon = 2 
)

# Group 2: GPT 2023 (old)
true_mu_gpt2023 <- c(
  prob_stay_win_raw = qnorm(0.65),    # much lower than human in PPC
  prob_stay_lose_raw = qnorm(0.50),   # all over the place in PPC (50/50)
  epsilon_raw = qnorm(0.19)           # some exploration
)
true_lambda_gpt2023 <- c( #slightly more precise than humans? but otherwise nearly the same
  prob_stay_win = 2,
  prob_stay_lose = 2,
  epsilon = 3
)

# Group 3: GPT 2025 (new)
true_mu_gpt2025 <- c(
  prob_stay_win_raw = qnorm(0.92),    # very high!!!
  prob_stay_lose_raw = qnorm(0.25),   # very unlikely to stay after a loss
  epsilon_raw = qnorm(0.05)           # very little exploration
)
true_lambda_gpt2025 <- c( # a lot more precise, especially in terms of exploration
  prob_stay_win = 4,
  prob_stay_lose = 4,
  epsilon = 10 #might be very tough to recover, given its low value and low variability!!!
)

cat("=== TRUE GROUP PARAMETERS ===\n")
cat("\nHumans:\n")
cat(sprintf("  mu: prob_stay_win=%.3f (p=%.3f), prob_stay_lose=%.3f (p=%.3f), epsilon=%.3f (p=%.3f)\n",
            true_mu_human[1], pnorm(true_mu_human[1]),
            true_mu_human[2], pnorm(true_mu_human[2]),
            true_mu_human[3], pnorm(true_mu_human[3])))
cat(sprintf("  lambda: %.1f, %.1f, %.1f\n", true_lambda_human[1], true_lambda_human[2], true_lambda_human[3]))

cat("\nGPT 2023:\n")
cat(sprintf("  mu: prob_stay_win=%.3f (p=%.3f), prob_stay_lose=%.3f (p=%.3f), epsilon=%.3f (p=%.3f)\n",
            true_mu_gpt2023[1], pnorm(true_mu_gpt2023[1]),
            true_mu_gpt2023[2], pnorm(true_mu_gpt2023[2]),
            true_mu_gpt2023[3], pnorm(true_mu_gpt2023[3])))
cat(sprintf("  lambda: %.1f, %.1f, %.1f\n", true_lambda_gpt2023[1], true_lambda_gpt2023[2], true_lambda_gpt2023[3]))

cat("\nGPT 2025:\n")
cat(sprintf("  mu: prob_stay_win=%.3f (p=%.3f), prob_stay_lose=%.3f (p=%.3f), epsilon=%.3f (p=%.3f)\n",
            true_mu_gpt2025[1], pnorm(true_mu_gpt2025[1]),
            true_mu_gpt2025[2], pnorm(true_mu_gpt2025[2]),
            true_mu_gpt2025[3], pnorm(true_mu_gpt2025[3])))
cat(sprintf("  lambda: %.1f, %.1f, %.1f\n", true_lambda_gpt2025[1], true_lambda_gpt2025[2], true_lambda_gpt2025[3]))

# ---------------------------
# STORAGE FOR RESULTS
# ---------------------------
results <- data.frame(
  sim_id = integer(),
  group = character(),
  
  # True values
  true_mu_prob_stay_win = numeric(),
  true_mu_prob_stay_lose = numeric(),
  true_mu_epsilon = numeric(),
  true_lambda_prob_stay_win = numeric(),
  true_lambda_prob_stay_lose = numeric(),
  true_lambda_epsilon = numeric(),
  
  # Recovered values
  rec_mu_prob_stay_win = numeric(),
  rec_mu_prob_stay_lose = numeric(),
  rec_mu_epsilon = numeric(),
  rec_lambda_prob_stay_win = numeric(),
  rec_lambda_prob_stay_lose = numeric(),
  rec_lambda_epsilon = numeric(),
  
  # Convergence
  max_rhat = numeric(),
  min_ess = numeric(),
  
  stringsAsFactors = FALSE
)

# Storage for group differences (contrasts)
contrasts <- data.frame(
  sim_id = integer(),
  parameter = character(),
  
  # True differences
  true_diff_gpt2023_human = numeric(),
  true_diff_gpt2025_human = numeric(),
  true_diff_gpt2025_gpt2023 = numeric(),
  
  # Recovered differences
  rec_diff_gpt2023_human = numeric(),
  rec_diff_gpt2025_human = numeric(),
  rec_diff_gpt2025_gpt2023 = numeric(),
  
  stringsAsFactors = FALSE
)

# Storage for hyperparameters
hyperparams <- data.frame(
  sim_id = integer(),
  group = character(),
  
  mu_prob_stay_win = numeric(),
  mu_prob_stay_lose = numeric(),
  mu_epsilon = numeric(),
  lambda_prob_stay_win = numeric(),
  lambda_prob_stay_lose = numeric(),
  lambda_epsilon = numeric(),
  
  stringsAsFactors = FALSE
)

# ============================================================================
# SIMULATION LOOP
# ============================================================================
start_time <- Sys.time()
cat("Simulation started: ", start_time)

for (sim in 1:n_simulations) {
  
  cat(sprintf("\n========== SIMULATION %d/%d ==========\n", sim, n_simulations))
  
  # ---------------------------
  # SIMULATE DATA
  # ---------------------------
  all_x <- list()
  all_X <- list()
  all_group <- c()
  
  for (g in 1:3) { 
    if (g == 1) {
      mu_g <- true_mu_human
      lambda_g <- true_lambda_human
      group_name <- "human"
    } else if (g == 2) {
      mu_g <- true_mu_gpt2023
      lambda_g <- true_lambda_gpt2023
      group_name <- "gpt2023"
    } else {
      mu_g <- true_mu_gpt2025
      lambda_g <- true_lambda_gpt2025
      group_name <- "gpt2025"
    }
    
    # Store hyperparameters (make sure they are readable, probabilities)
    hyperparams <- rbind(hyperparams, data.frame(
      sim_id = sim,
      group = group_name,
      
      mu_prob_stay_win = pnorm(mu_g[1]),
      mu_prob_stay_lose = pnorm(mu_g[2]),
      mu_epsilon = pnorm(mu_g[3]),
      lambda_prob_stay_win = lambda_g[1],
      lambda_prob_stay_lose = lambda_g[2],
      lambda_epsilon = lambda_g[3]
    ))
    
    # Sample individual parameters from group distribution
    prob_stay_win_sub <- pnorm(rnorm(n_subjects_per_group, mu_g[1], 1/sqrt(lambda_g[1])))
    prob_stay_lose_sub <- pnorm(rnorm(n_subjects_per_group, mu_g[2], 1/sqrt(lambda_g[2])))
    epsilon_sub <- pnorm(rnorm(n_subjects_per_group, mu_g[3], 1/sqrt(lambda_g[3])))
    
    # Simulate choices
    for (s in 1:n_subjects_per_group) {
      x_s <- rep(NA, n_trials)
      X_s <- rep(NA, n_trials)
      x_s[1] <- sample(1:4, 1)
      p_win <- c(0.5, 0.5, 0.5, 0.5)
      X_s[1] <- rbinom(1, 1, p_win[x_s[1]])
      
      for (t in 2:n_trials) {
        prev_choice <- x_s[t-1]
        prev_outcome <- X_s[t-1]
        
        if (runif(1) < epsilon_sub[s]) {
          x_s[t] <- sample(1:4, 1)
        } else {
          if (prev_outcome == 1) {
            x_s[t] <- ifelse(runif(1) < prob_stay_win_sub[s], prev_choice, sample(setdiff(1:4, prev_choice), 1))
          } else {
            x_s[t] <- ifelse(runif(1) < prob_stay_lose_sub[s], prev_choice, sample(setdiff(1:4, prev_choice), 1))
          }
        }
        
        X_s[t] <- rbinom(1, 1, p_win[x_s[t]])
      }
      
      all_x[[length(all_x) + 1]] <- x_s
      all_X[[length(all_X) + 1]] <- X_s
      all_group <- c(all_group, g)
    }
  }
  
  x_mat <- do.call(rbind, all_x)
  X_mat <- do.call(rbind, all_X)
  
  # ---------------------------
  # FIT MODEL
  # ---------------------------
  jags_data <- list(
    x = x_mat,
    X = X_mat,
    ntrials = rep(n_trials, nrow(x_mat)),
    nsubs = nrow(x_mat),
    group = all_group,
    n_groups = 3
  )
  
  jags_params <- c(
    "mu_prob_stay_win_raw", "mu_prob_stay_lose_raw", "mu_epsilon_raw",
    "lambda_prob_stay_win", "lambda_prob_stay_lose", "lambda_epsilon"
  )
  
  cat("Running JAGS...\n")
  samples <- jags.parallel(
    data = jags_data,
    inits = NULL,
    parameters.to.save = jags_params,
    model.file = model_file,
    n.chains = 3,
    n.iter = 5000,
    n.burnin = 1000,
    n.thin = 1,
    n.cluster = 3,
    jags.seed = 1234
  )
  
  # ---------------------------
  # EXTRACT RESULTS
  # ---------------------------
  sims <- samples$BUGSoutput$sims.list
  summary_mat <- samples$BUGSoutput$summary
  
  param_pattern <- paste0("^(", paste(jags_params, collapse="|"), ")\\[")
  param_rows <- grep(param_pattern, rownames(summary_mat))
  
  rhat_vec <- summary_mat[param_rows, "Rhat"]
  ess_vec <- summary_mat[param_rows, "n.eff"]
  max_rhat <- max(rhat_vec, na.rm = TRUE)
  min_ess <- min(ess_vec, na.rm = TRUE)
  
  cat("\nConvergence diagnostics (Rhat and ESS):\n")
  for (i in param_rows) {
    cat(sprintf("  %s: Rhat=%.3f, ESS=%.0f\n", rownames(summary_mat)[i], summary_mat[i,"Rhat"], summary_mat[i,"n.eff"]))
  }
  if (max_rhat > 1.1) cat("Convergence warning: max Rhat > 1.1\n")
  
  cat(sprintf("Convergence: max Rhat = %.3f, min ESS = %.0f\n", max_rhat, min_ess))
  
  # ---------------------------
  # Extract group-level parameters
  # ---------------------------
  for (g in 1:3) {
    if (g == 1) {true_mu <- true_mu_human; true_lambda <- true_lambda_human; group_name <- "human"}
    if (g == 2) {true_mu <- true_mu_gpt2023; true_lambda <- true_lambda_gpt2023; group_name <- "gpt2023"}
    if (g == 3) {true_mu <- true_mu_gpt2025; true_lambda <- true_lambda_gpt2025; group_name <- "gpt2025"}
    
    results <- rbind(results, data.frame(
      sim_id = sim,
      group = group_name,
      true_mu_prob_stay_win = true_mu[1],
      true_mu_prob_stay_lose = true_mu[2],
      true_mu_epsilon = true_mu[3],
      true_lambda_prob_stay_win = true_lambda[1],
      true_lambda_prob_stay_lose = true_lambda[2],
      true_lambda_epsilon = true_lambda[3],
      rec_mu_prob_stay_win = median(sims$mu_prob_stay_win_raw[,g]),
      rec_mu_prob_stay_lose = median(sims$mu_prob_stay_lose_raw[,g]),
      rec_mu_epsilon = median(sims$mu_epsilon_raw[,g]),
      rec_lambda_prob_stay_win = median(sims$lambda_prob_stay_win[,g]),
      rec_lambda_prob_stay_lose = median(sims$lambda_prob_stay_lose[,g]),
      rec_lambda_epsilon = median(sims$lambda_epsilon[,g]),
      max_rhat = max_rhat,
      min_ess = min_ess
    ))
  }
  
  # ---------------------------
  # Extract group contrasts
  # ---------------------------
  for (param in c("prob_stay_win", "prob_stay_lose", "epsilon")) {
    param_name <- paste0("mu_", param, "_raw")
    rec_human <- MPD(sims[[param_name]][,1])
    rec_gpt2023 <- MPD(sims[[param_name]][,2])
    rec_gpt2025 <- MPD(sims[[param_name]][,3])
    
    if (param == "prob_stay_win") {true_human <- true_mu_human[1]; true_gpt2023 <- true_mu_gpt2023[1]; true_gpt2025 <- true_mu_gpt2025[1]}
    if (param == "prob_stay_lose") {true_human <- true_mu_human[2]; true_gpt2023 <- true_mu_gpt2023[2]; true_gpt2025 <- true_mu_gpt2025[2]}
    if (param == "epsilon") {true_human <- true_mu_human[3]; true_gpt2023 <- true_mu_gpt2023[3]; true_gpt2025 <- true_mu_gpt2025[3]}
    
    contrasts <- rbind(contrasts, data.frame(
      sim_id = sim,
      parameter = param,
      true_diff_gpt2023_human = true_gpt2023 - true_human,
      true_diff_gpt2025_human = true_gpt2025 - true_human,
      true_diff_gpt2025_gpt2023 = true_gpt2025 - true_gpt2023,
      rec_diff_gpt2023_human = rec_gpt2023 - rec_human,
      rec_diff_gpt2025_human = rec_gpt2025 - rec_human,
      rec_diff_gpt2025_gpt2023 = rec_gpt2025 - rec_gpt2023
    ))
  }
  
  # ---------------------------
  # Save incremental results
  # ---------------------------
  write.csv(results, file.path(results_dir, "group_params_recovery.csv"), row.names = FALSE)
  write.csv(contrasts, file.path(results_dir, "group_contrasts_recovery.csv"), row.names = FALSE)
  write.csv(hyperparams, file.path(results_dir, "group_hyperparams.csv"), row.names = FALSE)
  
  # Progress update
  elapsed <- as.numeric(difftime(Sys.time(), start_time, units="mins"))
  avg_time <- elapsed / sim
  est_remaining <- avg_time * (n_simulations - sim)
  cat(sprintf("Progress: %d/%d | Elapsed: %.1f min | Est. remaining: %.1f min\n",
              sim, n_simulations, elapsed, est_remaining))
}

# ============================================================================
# FINAL ANALYSIS & PLOTS
# ============================================================================

end_time <- Sys.time()
total_time <- difftime(end_time, start_time, units="mins")
cat(sprintf("\n=== COMPLETED in %.2f minutes ===\n", total_time))

# ---------------------------
# GROUP PARAMETER RECOVERY METRICS
# ---------------------------
cat("\n=== GROUP PARAMETER RECOVERY ===\n")
for (g_name in c("human", "gpt2023", "gpt2025")) {
  cat(sprintf("\n%s:\n", toupper(g_name)))
  res_g <- results[results$group == g_name, ]
  
  for (param in c("prob_stay_win", "prob_stay_lose", "epsilon")) {
    true_col <- paste0("true_mu_", param)
    rec_col <- paste0("rec_mu_", param)
    
    bias <- mean(res_g[[rec_col]] - res_g[[true_col]], na.rm=TRUE)
    rmse <- sqrt(mean((res_g[[rec_col]] - res_g[[true_col]])^2, na.rm=TRUE))
    
    cat(sprintf("  %s: bias=%.3f, RMSE=%.3f\n", param, bias, rmse))
  }
}

# ---------------------------
# GROUP CONTRAST RECOVERY METRICS
# ---------------------------
cat("\n=== GROUP CONTRAST RECOVERY ===\n")
for (param in c("prob_stay_win", "prob_stay_lose", "epsilon")) {
  cat(sprintf("\n%s:\n", param))
  con_p <- contrasts[contrasts$parameter == param, ]
  
  for (contrast in c("gpt2023_human", "gpt2025_human", "gpt2025_gpt2023")) {
    true_col <- paste0("true_diff_", contrast)
    rec_col <- paste0("rec_diff_", contrast)
    
    r <- cor(con_p[[true_col]], con_p[[rec_col]], use="complete.obs")
    bias <- mean(con_p[[rec_col]] - con_p[[true_col]], na.rm=TRUE)
    rmse <- sqrt(mean((con_p[[rec_col]] - con_p[[true_col]])^2, na.rm=TRUE))
    
    cat(sprintf("  %s: bias=%.3f, r=%.3f, RMSE=%.3f\n", contrast, bias, r, rmse))
  }
}

# ============================================================================
# PLOTS - GROUP PARAMETERS (one file per group)
# ============================================================================
for (g_name in c("human", "gpt2023", "gpt2025")) {
  res_g <- results[results$group == g_name, ]
  hyper_g <- hyperparams[hyperparams$group == g_name, ]
  
  png(file.path(results_dir, paste0("recovery_group_params_", g_name, ".png")), 
      width = 1200, height = 1200, res = 150)
  
  par(mfrow=c(1,3), mar=c(4,4,3,1))
  
  for (param in c("prob_stay_win", "prob_stay_lose", "epsilon")) {
    true_col <- paste0("true_mu_", param)
    rec_col <- paste0("rec_mu_", param)
    
    plot(res_g[[true_col]], res_g[[rec_col]],
         main = param,
         xlab = "True", ylab = "Recovered",
         pch = 20, col = rgb(0,0,0,0.5))
    abline(a=0, b=1, col="red", lwd=2)
    abline(h=mean(hyper_g[[paste0("mu_", param)]]), col="blue", lty=2)
    
    #might be SD 0, so NA, might have to omit here
    r <- cor(res_g[[true_col]], res_g[[rec_col]], use="complete.obs")
    legend("topleft", legend = sprintf("r = %.3f", r), bty="n")
  }
  
  dev.off()
}

# ============================================================================
# PLOTS - GROUP CONTRASTS
# ============================================================================
png(file.path(results_dir, "recovery_group_contrasts_all_sims.png"),
    width = 2400, height = 800, res = 150)
par(mfrow=c(1,3), mar=c(4,4,3,1))

for (param in c("prob_stay_win", "prob_stay_lose", "epsilon")) {
  con_p <- contrasts[contrasts$parameter == param, ]
  
  all_true <- c(con_p$true_diff_gpt2023_human,
                con_p$true_diff_gpt2025_human,
                con_p$true_diff_gpt2025_gpt2023)
  all_rec <- c(con_p$rec_diff_gpt2023_human,
               con_p$rec_diff_gpt2025_human,
               con_p$rec_diff_gpt2025_gpt2023)
  
  plot(all_true, all_rec,
       main = paste("Contrasts:", param),
       xlab = "True Difference", ylab = "Recovered Difference",
       pch = 20, col = rgb(0,0,0,0.3))
  abline(a=0, b=1, col="red", lwd=2)
  abline(h=0, v=0, col="gray", lty=2)
  
  r <- cor(all_true, all_rec, use="complete.obs")
  legend("topleft", legend = sprintf("r = %.3f", r), bty="n")
}

dev.off()

# ============================================================================
# EXTRA PLOTS - POSTERIOR DENSITIES ACROSS ALL SIMULATIONS
# ============================================================================
params <- c("mu_prob_stay_win_raw", "mu_prob_stay_lose_raw", "mu_epsilon_raw")
groups <- c("human", "gpt2023", "gpt2025")

plot_data <- data.frame()
for (sim in 1:n_simulations) {
  sims_list <- samples$BUGSoutput$sims.list
  for (param in params) {
    for (g in 1:3) {
      plot_data <- rbind(plot_data, data.frame(
        value = sims_list[[param]][, g],
        group = groups[g],
        parameter = param,
        sim = sim
      ))
    }
  }
}

plot_data$parameter <- factor(plot_data$parameter,
                              levels = params,
                              labels = c("Prob Stay Win", "Prob Stay Lose", "Epsilon"))

vline_data <- data.frame(
  parameter = rep(c("Prob Stay Win", "Prob Stay Lose", "Epsilon"), 3),
  group = rep(groups, each = 3),
  xintercept = c(true_mu_human, true_mu_gpt2023, true_mu_gpt2025)
)

p <- ggplot(plot_data, aes(x = value, fill = group, color = group)) +
  geom_density(alpha = 0.2) +
  facet_wrap(~parameter, scales = "free") +
  geom_vline(data = vline_data, aes(xintercept = xintercept, color = group), 
             linetype = "dashed", alpha = 0.7) +
  labs(title = sprintf("Posterior densities across all %d simulations", n_simulations),
       x = "Parameter value", y = "Density") +
  theme_minimal() +
  theme(legend.position = "top")

ggsave(filename = file.path(results_dir, "posterior_densities_all_sims.png"),
       plot = p, width = 12, height = 6, dpi = 150)

cat("\nPosterior density plot for all simulations saved to:", results_dir, "\n")
cat("All done!\n")