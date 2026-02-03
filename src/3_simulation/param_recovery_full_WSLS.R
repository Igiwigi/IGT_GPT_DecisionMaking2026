# ============================================================================
# GROUP & INDIV PARAMETER RECOVERY - HIERARCHICAL WSLS (win-stay / lose-stay + epsilon)
# ============================================================================
#need to report corr (true vs recovered), bias, slope (they're included, but only at the end right now)
#a bit of a mess unfortunately (could have done this cleaner, such as by making the plots in a separate file)

library(pacman)
pacman::p_load(extraDistr, R2jags, parallel, jsonlite)

# setup
set.seed(1234)
root_path <- "~/IGT_LLM_DecisionMaking"
filename <- "simulated_group_winlose.json"
file <- file.path(root_path, "src/3_simulation/recovery/simulated_data", filename)

# Load JSON simulated group data
data <- jsonlite::fromJSON(file)

#using median not mode!!

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

# Individual-level parameter storage
max_nsubs <- max(sapply(data$x, nrow))
indiv_params <- c("prob_stay_win", "prob_stay_lose", "epsilon")  # For JAGS output
indiv_params_json <- c("prob_stay_win_sub", "prob_stay_lose_sub", "epsilon_sub")  # From JSON
true_indiv <- array(NA, dim=c(n_groups, max_nsubs, length(indiv_params)))
infer_indiv <- array(NA, dim=c(n_groups, max_nsubs, length(indiv_params)))

#rhat param storage
group_params <- c(params, lambda_params)
rhat_vals <- matrix(NA, nrow=n_groups, ncol=length(group_params))
colnames(rhat_vals) <- paste0("Rhat_", group_params)

# Convergence diagnostics storage
max_rhat <- rep(NA, n_groups)
min_ess <- rep(NA, n_groups)
n_subjects <- rep(NA, n_groups)

# output paths
results_dir <- file.path(root_path, "src/3_simulation/recovery/results")
csv_file <- file.path(results_dir, "param_recovery_group_WSLS_incremental.csv")
csv_indiv_file <- file.path(results_dir, "param_recovery_indiv_WSLS_incremental.csv")
plot_mu_file <- file.path(results_dir, "recovery_mu_WSLS_incremental.png")
plot_lambda_file <- file.path(results_dir, "recovery_lambda_WSLS_incremental.png")
plot_indiv_file <- file.path(results_dir, "recovery_indiv_WSLS_incremental.png")

# ---------------------------
# LOOP OVER GROUPS
# ---------------------------
for (i in 1:n_groups) {
  cat(sprintf("\n=== Processing group %d/%d ===\n", i, n_groups))
  
  # extract choices and outcomes (from simulated data)
  x <- data$x[[i]] + 1
  X <- data$X[[i]]
  nsubs <- nrow(x)
  ntrials <- rep(ncol(x), nsubs)
  n_subjects[i] <- nsubs
  
  # extract true group parameters
  mu_prob_stay_win_raw <- as.numeric(data$mu_prob_stay_win_raw[i])
  mu_prob_stay_lose_raw <- as.numeric(data$mu_prob_stay_lose_raw[i])
  mu_epsilon_raw <- as.numeric(data$mu_epsilon_raw[i])
  
  lambda_prob_stay_win <- as.numeric(data$lambda_prob_stay_win[i])
  lambda_prob_stay_lose <- as.numeric(data$lambda_prob_stay_lose[i])
  lambda_epsilon <- as.numeric(data$lambda_epsilon[i])
  
  # extract true individual parameters
  for(p_idx in 1:length(indiv_params)) {
    p_json <- indiv_params_json[p_idx]  # Use _sub suffix for JSON
    param_values <- data[[p_json]][[i]]
    
    # Handle different possible structures
    if(is.null(param_values)) {
      warning(sprintf("Group %d: Parameter %s is NULL", i, p_json))
      next
    }
    
    # Convert to vector if needed
    if(is.matrix(param_values)) {
      param_values <- as.vector(t(param_values))  # transpose then vectorize
    } else if(is.data.frame(param_values)) {
      param_values <- unlist(param_values)
    } else if(is.list(param_values)) {
      param_values <- unlist(param_values)
    }
    
    # Ensure right length
    if(length(param_values) < nsubs) {
      warning(sprintf("Group %d: Parameter %s has length %d but nsubs=%d", 
                      i, p_json, length(param_values), nsubs))
      next
    }
    
    true_indiv[i, 1:nsubs, p_idx] <- param_values[1:nsubs]
  }
  
  # prepare JAGS data
  jags_data <- list(
    x = x,
    X = X,
    ntrials = ntrials,
    nsubs = nsubs
  )
  
  # parameters to monitor
  jags_params <- c(params, lambda_params, indiv_params)
  model_file <- file.path(root_path, "models", "simpler_models", "WSLS_hier.txt")
  
  # run JAGS
  cat("Running JAGS ...\n")
  group_start <- Sys.time()
  
  samples <- jags.parallel(
    data = jags_data,
    inits = NULL,
    parameters.to.save = jags_params,
    model.file = model_file,
    n.chains = 3,
    n.iter = 5000,
    n.burnin = 1000,
    n.thin = 1,
    n.cluster = 4,
    jags.seed= 1234
  )
  
  # ---------------------------
  # CONVERGENCE DIAGNOSTICS (DETAILED)
  # ---------------------------
  summary_mat <- samples$BUGSoutput$summary
  conv_params <- c(params, lambda_params, indiv_params)
  conv_params <- conv_params[conv_params %in% rownames(summary_mat)]
  rhat_vec <- summary_mat[conv_params, "Rhat"]
  ess_vec  <- summary_mat[conv_params, "n.eff"]
  max_rhat[i] <- max(rhat_vec, na.rm = TRUE)
  min_ess[i]  <- min(ess_vec,  na.rm = TRUE)
  
  #some prints for each run
  cat("Rhat by parameter and group:\n")
  group_labels <- c("human", "gpt2023", "gpt2025")
  
  for (p in jags_params) {
    cat(sprintf("  %s:\n", p))
    
    for (g in 1:3) {
      row_name <- paste0(p, "[", g, "]")
      
      if (row_name %in% rownames(summary_mat)) {
        rhat_val <- summary_mat[row_name, "Rhat"]
        ess_val  <- summary_mat[row_name, "n.eff"]
        
        cat(sprintf(
          "    %-8s  Rhat = %.3f | ESS = %.0f\n",
          group_labels[g], rhat_val, ess_val
        ))
      }
    }
  }
  
  cat("\nConvergence diagnostics:\n")
  print(summary_mat[conv_params, c("Rhat", "n.eff")], digits = 3)
  
  group_end <- Sys.time()
  cat(sprintf("Group %d completed in %.2f minutes\n", i, as.numeric(difftime(group_end, group_start, units="mins"))))
  
  sims <- samples$BUGSoutput$sims.list
  
  # store true and inferred group means (using median instead of MPD)
  true_mu[i, ] <- c(mu_prob_stay_win_raw, mu_prob_stay_lose_raw, mu_epsilon_raw)
  infer_mu[i, ] <- sapply(params, function(p) median(sims[[p]]))
  
  # store true and inferred group lambdas (using median instead of MPD)
  true_lambda[i, ] <- c(lambda_prob_stay_win, lambda_prob_stay_lose, lambda_epsilon)
  infer_lambda[i, ] <- sapply(lambda_params, function(p) median(sims[[p]]))
  
  # store inferred individual parameters (using median instead of MPD)
  for(p_idx in 1:length(indiv_params)) {
    p <- indiv_params[p_idx]
    infer_indiv[i, 1:nsubs, p_idx] <- apply(sims[[p]], 2, median)
  }
  
  if(max_rhat[i] > 1.1) {
    warning(sprintf("Group %d: Poor convergence (Rhat = %.3f)\n", i, max_rhat[i]))
  }
  
  # ---------------------------
  # SAVE INCREMENTAL RESULTS - GROUP LEVEL
  # ---------------------------
  
  #rhat for each grp
  for (p_idx in seq_along(group_params)) {
    p <- group_params[p_idx]
    if (p %in% rownames(summary_mat)) {
      rhat_vals[i, p_idx] <- summary_mat[p, "Rhat"]
    } else {
      rhat_vals[i, p_idx] <- NA
    }
  }
  
  #dataframe itself
  df_group <- data.frame(
    group_id = 1:n_groups,
    nsubs = n_subjects,
    max_rhat = max_rhat,
    min_ess = min_ess,
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
    infer_lambda_epsilon = infer_lambda[,3],
    rhat_vals #rhat vals
  )
  
  
  write.csv(df_group, csv_file, row.names = FALSE)
  cat(sprintf("Saved group-level results to %s\n", basename(csv_file)))
  
  # ---------------------------
  # SAVE INCREMENTAL RESULTS - INDIVIDUAL LEVEL
  # ---------------------------
  df_indiv_list <- list()
  for(g in 1:n_groups) {
    if(!is.na(n_subjects[g]) && n_subjects[g] > 0) {
      for(s in 1:n_subjects[g]) {
        df_indiv_list[[length(df_indiv_list) + 1]] <- data.frame(
          group_id = g,
          subject_id = s,
          true_prob_stay_win = true_indiv[g, s, 1],
          true_prob_stay_lose = true_indiv[g, s, 2],
          true_epsilon = true_indiv[g, s, 3],
          infer_prob_stay_win = infer_indiv[g, s, 1],
          infer_prob_stay_lose = infer_indiv[g, s, 2],
          infer_epsilon = infer_indiv[g, s, 3]
        )
      }
    }
  }
  
  if(length(df_indiv_list) > 0) {
    df_indiv <- do.call(rbind, df_indiv_list)
    write.csv(df_indiv, csv_indiv_file, row.names = FALSE)
    cat(sprintf("Saved individual-level results to %s\n", basename(csv_indiv_file)))
  }
  
  # ---------------------------
  # UPDATE PLOTS
  # ---------------------------
  completed <- !is.na(infer_mu[,1])
  n_completed <- sum(completed)
  
  if(n_completed >= 2) {
    # plot mu parameters
    png(plot_mu_file, width = 1800, height = 600, res = 150)
    par(mfrow=c(1,3), mar=c(3,3,2,1))
    for(p in c("prob_stay_win_raw", "prob_stay_lose_raw", "epsilon_raw")) {
      true_vals <- df_group[[paste0("true_mu_", p)]][completed]
      infer_vals <- df_group[[paste0("infer_mu_", p)]][completed]
      plot(true_vals, infer_vals, main=p, xlab="True", ylab="Inferred")
      abline(a=0, b=1, col="red")
      if(n_completed >= 3) {
        r <- cor(true_vals, infer_vals, use="complete.obs")
        legend("topleft", legend=sprintf("r=%.3f\nn=%d", r, n_completed), bty="n", cex=0.8)
      } else {
        legend("topleft", legend=sprintf("n=%d", n_completed), bty="n", cex=0.8)
      }
    }
    dev.off()
    
    # plot lambda parameters
    png(plot_lambda_file, width = 1800, height = 600, res = 150)
    par(mfrow=c(1,3), mar=c(3,3,2,1))
    for(p in c("prob_stay_win", "prob_stay_lose", "epsilon")) {
      true_vals <- df_group[[paste0("true_lambda_", p)]][completed]
      infer_vals <- df_group[[paste0("infer_lambda_", p)]][completed]
      plot(true_vals, infer_vals, main=p, xlab="True", ylab="Inferred")
      abline(a=0, b=1, col="red")
      if(n_completed >= 3) {
        r <- cor(true_vals, infer_vals, use="complete.obs")
        legend("topleft", legend=sprintf("r=%.3f\nn=%d", r, n_completed), bty="n", cex=0.8)
      } else {
        legend("topleft", legend=sprintf("n=%d", n_completed), bty="n", cex=0.8)
      }
    }
    dev.off()
    
    # plot individual parameters
    if(length(df_indiv_list) > 0) {
      df_indiv_complete <- df_indiv[complete.cases(df_indiv[,3:8]), ]
      
      if(nrow(df_indiv_complete) >= 10) {
        png(plot_indiv_file, width = 1800, height = 600, res = 150)
        par(mfrow=c(1,3), mar=c(3,3,2,1))
        for(p in c("prob_stay_win", "prob_stay_lose", "epsilon")) {
          true_vals <- df_indiv_complete[[paste0("true_", p)]]
          infer_vals <- df_indiv_complete[[paste0("infer_", p)]]
          plot(true_vals, infer_vals, main=paste("Individual", p), 
               xlab="True", ylab="Inferred", pch=20, cex=0.5)
          abline(a=0, b=1, col="red")
          r <- cor(true_vals, infer_vals, use="complete.obs")
          mtext(sprintf("r=%.3f, n=%d", r, nrow(df_indiv_complete)), 
                side=3, line=0.5, adj=0, cex=0.7)
        }
        dev.off()
      }
    }
    
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
cat(sprintf("Group-level results: %s\n", csv_file))
cat(sprintf("Individual-level results: %s\n", csv_indiv_file))

# Correlations - Group level
completed <- !is.na(infer_mu[,1])
df_complete <- df_group[completed,]

# ---------------------------
# GROUP-LEVEL RECOVERY METRICS (r, slope, bias, RMSE)
# ---------------------------
for(p in c("prob_stay_win_raw","prob_stay_lose_raw","epsilon_raw")) {
  true_v  <- df_complete[[paste0("true_mu_", p)]]
  infer_v <- df_complete[[paste0("infer_mu_", p)]]
  
  df_group[[paste0("mu_", p, "_r")]]     <- NA
  df_group[[paste0("mu_", p, "_bias")]]  <- NA
  df_group[[paste0("mu_", p, "_rmse")]]  <- NA
  df_group[[paste0("mu_", p, "_slope")]] <- NA
  
  idx <- which(completed)
  df_group[idx, paste0("mu_", p, "_r")]     <- cor(true_v, infer_v, use="complete.obs")
  df_group[idx, paste0("mu_", p, "_bias")]  <- mean(infer_v - true_v, na.rm=TRUE)
  df_group[idx, paste0("mu_", p, "_rmse")]  <- sqrt(mean((infer_v - true_v)^2, na.rm=TRUE))
  df_group[idx, paste0("mu_", p, "_slope")] <- coef(lm(infer_v ~ true_v))[2]
}

for(p in c("prob_stay_win","prob_stay_lose","epsilon")) {
  true_v  <- df_complete[[paste0("true_lambda_", p)]]
  infer_v <- df_complete[[paste0("infer_lambda_", p)]]
  
  df_group[[paste0("lambda_", p, "_r")]]     <- NA
  df_group[[paste0("lambda_", p, "_bias")]]  <- NA
  df_group[[paste0("lambda_", p, "_rmse")]]  <- NA
  df_group[[paste0("lambda_", p, "_slope")]] <- NA
  
  idx <- which(completed)
  df_group[idx, paste0("lambda_", p, "_r")]     <- cor(true_v, infer_v, use="complete.obs")
  df_group[idx, paste0("lambda_", p, "_bias")]  <- mean(infer_v - true_v, na.rm=TRUE)
  df_group[idx, paste0("lambda_", p, "_rmse")]  <- sqrt(mean((infer_v - true_v)^2, na.rm=TRUE))
  df_group[idx, paste0("lambda_", p, "_slope")] <- coef(lm(infer_v ~ true_v))[2]
}

# overwrite group CSV with metrics
write.csv(df_group, csv_file, row.names = FALSE)
cat(sprintf("Updated group CSV with recovery metrics: %s\n", basename(csv_file)))

# ---------------------------
# INDIVIDUAL-LEVEL RECOVERY METRICS
# ---------------------------
if(exists("df_indiv") && nrow(df_indiv) > 0) {
  df_indiv_complete <- df_indiv[complete.cases(df_indiv[,3:8]), ]
  
  df_indiv_summary <- data.frame()
  for(p in c("prob_stay_win","prob_stay_lose","epsilon")) {
    true_v  <- df_indiv_complete[[paste0("true_", p)]]
    infer_v <- df_indiv_complete[[paste0("infer_", p)]]
    
    df_indiv_summary <- rbind(
      df_indiv_summary,
      data.frame(
        parameter = p,
        r     = cor(true_v, infer_v, use="complete.obs"),
        slope = coef(lm(infer_v ~ true_v))[2],
        bias  = mean(infer_v - true_v, na.rm=TRUE),
        rmse  = sqrt(mean((infer_v - true_v)^2, na.rm=TRUE))
      )
    )
  }
  
  write.csv(
    df_indiv_summary,
    file.path(results_dir, "param_recovery_indiv_summary_WSLS.csv"),
    row.names = FALSE
  )
  cat(sprintf("Saved individual-level summary metrics: %s\n", "param_recovery_indiv_summary_WSLS.csv"))
}

# ---------------------------
# CONVERGENCE DIAGNOSTICS SUMMARY
# ---------------------------
cat("\n=== CONVERGENCE DIAGNOSTICS ===\n")
cat(sprintf("Groups with Rhat > 1.1: %d/%d\n", 
            sum(max_rhat > 1.1, na.rm = TRUE), sum(!is.na(max_rhat)))) #63 have bad rhat, but the plots and vars are good, so fine?
cat(sprintf("Mean max Rhat: %.3f (SD=%.3f)\n", 
            mean(max_rhat, na.rm = TRUE), sd(max_rhat, na.rm = TRUE)))
cat(sprintf("Mean min ESS: %.0f (SD=%.0f)\n",
            mean(min_ess, na.rm = TRUE), sd(min_ess, na.rm = TRUE))) #low n for some of the vars

cat("\nDone!\n")


# ---------------------------
# CONVERGENCE DIAGNOSTICS SUMMARY EXTRA (RHAT & PARAMETER EXTREMENESS)
# ---------------------------
df_diag <- df_group

# Transform raw mus to probability scale
df_diag$true_prob_stay_win <- pnorm(df_diag$true_mu_prob_stay_win_raw)
df_diag$true_prob_stay_lose <- pnorm(df_diag$true_mu_prob_stay_lose_raw)
df_diag$true_epsilon_prob  <- pnorm(df_diag$true_mu_epsilon_raw)

# Flag problematic convergence
df_diag$bad_rhat <- df_diag$max_rhat > 1.1

# Summaries
cat("\nMean transformed parameters by Rhat group:\n")
print(aggregate(
  cbind(true_prob_stay_win, true_prob_stay_lose, true_epsilon_prob,
        Rhat_mu_prob_stay_win_raw, Rhat_mu_prob_stay_lose_raw, Rhat_mu_epsilon_raw,
        Rhat_lambda_prob_stay_win, Rhat_lambda_prob_stay_lose, Rhat_lambda_epsilon) ~ bad_rhat,
  data = df_diag,
  FUN = mean,
  na.rm = TRUE
))

cat("\nProportion of extreme values (p < .05 or > .95):\n")
extreme_percent <- function(x) {
  mean(x < 0.05 | x > 0.95, na.rm = TRUE) * 100
}

#amount of extreme variables, in percent (around 8-13%)
print(aggregate(
  cbind(true_prob_stay_win, true_prob_stay_lose, true_epsilon_prob) ~ bad_rhat,
  data = df_diag,
  FUN = extreme_percent
))