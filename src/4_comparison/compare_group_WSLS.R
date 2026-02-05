# ============================================================================
# GROUP COMPARE AFTER VETTING MODEL
# ============================================================================
# WSLS script to compare the WSLS parameters between the real groups (human, GPT 2023, GPT 2025)
# For reporting the differences between the main WSLS parameters: stay-win, stay-lose, epsilon
# Adapted the grp recovery script to real data (but only fitting once)
# Should have maybe made more of this pipeline into functions etc.

library(pacman)
pacman::p_load(R2jags, parallel, dplyr)
set.seed(1234)

probit_to_prob <- function(x) pnorm(x)

# ---------------------------
# PATHS & LOADING
# ---------------------------
root_path <- "~/IGT_LLM_DecisionMaking"
results_dir <- file.path(root_path, "src/4_comparison/results/group_parameters")
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
# DATA PREPARATION
# ---------------------------
n_trials <- 100
group_names <- c("ahn_hc", "gpt_OLD_2023", "gpt_NEW_2025")

# add grp label to each dataset
hc_data$group <- "ahn_hc"
old_gpt_data$group <- "gpt_OLD_2023"
new_gpt_data$group <- "gpt_NEW_2025"

# combine all data into one that can be passed to jags (model differentiates per group)
all_data <- rbind(hc_data, old_gpt_data, new_gpt_data)

# Get unique subjects per group
subjects_per_group <- all_data %>% 
  group_by(group) %>% 
  summarise(n_subjects = n_distinct(subjID))

print("Subjects per group:")
print(subjects_per_group) #newGPT has slightly more (32); since Bayesian, this is ok

#create the matrix for JAGS
unique_subjects <- unique(all_data$subjID)
n_subjects <- length(unique_subjects)

# Initialize matrices to fill later
x_mat <- matrix(NA, nrow = n_subjects, ncol = n_trials)
X_mat <- matrix(NA, nrow = n_subjects, ncol = n_trials)
group_idx <- numeric(n_subjects)

# Fill matrices
# x = deck choice (1,2,3,4), X = outcome (int, -/+), subjID = subject identifier
for(i in 1:n_subjects){
  subj_id <- unique_subjects[i]
  subj_data <- all_data[all_data$subjID == subj_id, ]
  
  n_trials_subj <- nrow(subj_data)
  x_mat[i, 1:n_trials_subj] <- subj_data$x #deck choice!!
  X_mat[i, 1:n_trials_subj] <- subj_data$X #outcome
  
  # group index (1=ahn_hc, 2=gpt_OLD_2023, 3=gpt_NEW_2025)
  group_idx[i] <- which(group_names == subj_data$group[1])
  print(paste("Subject", subj_id, "assigned to group", group_names[group_idx[i]]))
  print(group_idx[i])
}

# Convert outcomes to binary (win=1, lose=0)
X_binary <- ifelse(X_mat >= 0, 1, 0)

# ---------------------------
# PREPARE JAGS DATA
# ---------------------------
jags_data <- list(
  x = x_mat,
  X = X_binary,
  ntrials = apply(!is.na(x_mat), 1, sum), #number of trials per subject (that arent NA)
  nsubs = n_subjects,
  group = group_idx,
  n_groups = 3
)

jags_params <- c("mu_prob_stay_win_raw", "mu_prob_stay_lose_raw", "mu_epsilon_raw",
                 "lambda_prob_stay_win", "lambda_prob_stay_lose", "lambda_epsilon")

# ---------------------------
# RUN JAGS
# ---------------------------
cat("\n=== RUNNING HIERARCHICAL GROUP COMPARISON WITH REAL DATA ===\n")
cat("Total subjects:", n_subjects, "\n")
cat("Groups:", paste(group_names, collapse=", "), "\n")
cat("Analysis started:", format(Sys.time()), "\n\n")

start_time <- Sys.time()
samples <- jags.parallel(
  data = jags_data,
  inits = NULL,
  parameters.to.save = jags_params,
  model.file = model_file,
  n.chains = 3,
  n.iter = 10000, #i upped all this to have slightly better convergence overall (will also be reflected in all plots after this)
  n.burnin = 2000,
  n.thin = 1, #to remove autocorrelation when high n.iter, supposedly
  n.cluster = 4,
  jags.seed= 1234
)

end_time <- Sys.time()

cat(sprintf("\nCompleted in %.2f minutes\n", as.numeric(difftime(end_time, start_time, units="mins"))))

# ---------------------------
# EXTRACT RESULTS
# ---------------------------
summary_mat <- samples$BUGSoutput$summary
sims_list <- samples$BUGSoutput$sims.list

# Storage for results
group_params <- data.frame()
group_contrasts <- data.frame()

# ---------------------------
# Extract group-level parameters (trying to take only what is needed!)
# ---------------------------
cat("\n=== GROUP-LEVEL PARAMETER ESTIMATES ===\n")

for(g in 1:length(group_names)){
  group_name <- group_names[g]
  
  n_subjects = subjects_per_group$n_subjects[subjects_per_group$group == group_name] #save number of subjects
  
  # Extract posterior distributions (probit)
  mu_prob_stay_win_raw <- sims_list$mu_prob_stay_win_raw[, g]
  mu_prob_stay_lose_raw <- sims_list$mu_prob_stay_lose_raw[, g]
  mu_epsilon_raw <- sims_list$mu_epsilon_raw[, g]
  lambda_prob_stay_win <- sims_list$lambda_prob_stay_win[, g]
  lambda_prob_stay_lose <- sims_list$lambda_prob_stay_lose[, g]
  lambda_epsilon <- sims_list$lambda_epsilon[, g]
  
  # save in probability space, not probit
  mu_prob_stay_win_prob <- probit_to_prob(mu_prob_stay_win_raw)
  mu_prob_stay_lose_prob <- probit_to_prob(mu_prob_stay_lose_raw)
  mu_epsilon_prob <- probit_to_prob(mu_epsilon_raw)
  
  # Summary statistics
  group_params <- rbind(group_params, data.frame(
    group = group_name,
    
    # Probability scale (transformed to probability; not saving raw)
    mu_prob_stay_win_median = median(mu_prob_stay_win_prob),
    mu_prob_stay_win_q025 = quantile(mu_prob_stay_win_prob, 0.025),
    mu_prob_stay_win_q975 = quantile(mu_prob_stay_win_prob, 0.975),
    
    mu_prob_stay_lose_median = median(mu_prob_stay_lose_prob),
    mu_prob_stay_lose_q025 = quantile(mu_prob_stay_lose_prob, 0.025),
    mu_prob_stay_lose_q975 = quantile(mu_prob_stay_lose_prob, 0.975),
    
    mu_epsilon_median = median(mu_epsilon_prob),
    mu_epsilon_q025 = quantile(mu_epsilon_prob, 0.025),
    mu_epsilon_q975 = quantile(mu_epsilon_prob, 0.975),
    
    # Convergence diagnostics: rhat & effective size
    rhat_prob_stay_win = summary_mat[paste0("mu_prob_stay_win_raw[", g, "]"), "Rhat"],
    rhat_prob_stay_lose = summary_mat[paste0("mu_prob_stay_lose_raw[", g, "]"), "Rhat"],
    rhat_epsilon = summary_mat[paste0("mu_epsilon_raw[", g, "]"), "Rhat"],

    n_eff_prob_stay_win = summary_mat[paste0("mu_prob_stay_win_raw[", g, "]"), "n.eff"], 
    n_eff_prob_stay_lose = summary_mat[paste0("mu_prob_stay_lose_raw[", g, "]"), "n.eff"],
    n_eff_epsilon = summary_mat[paste0("mu_epsilon_raw[", g, "]"), "n.eff"]
  ))
  
  # Print summary
  cat(sprintf("\n%s:\n", toupper(group_name)))
  cat(sprintf("  P(stay|win): %.3f [%.3f, %.3f]\n", 
              median(mu_prob_stay_win_prob), 
              quantile(mu_prob_stay_win_prob, 0.025),
              quantile(mu_prob_stay_win_prob, 0.975)))
  cat(sprintf("  P(stay|lose): %.3f [%.3f, %.3f]\n", 
              median(mu_prob_stay_lose_prob),
              quantile(mu_prob_stay_lose_prob, 0.025),
              quantile(mu_prob_stay_lose_prob, 0.975)))
  cat(sprintf("  Epsilon: %.3f [%.3f, %.3f]\n", 
              median(mu_epsilon_prob),
              quantile(mu_epsilon_prob, 0.025),
              quantile(mu_epsilon_prob, 0.975)))
}

# ---------------------------
# Compute group contrasts
# ---------------------------
cat("\n=== GROUP CONTRASTS ===\n")

param_names <- c("prob_stay_win", "prob_stay_lose", "epsilon")
cat("\nComputing contrasts for parameters:", paste(param_names, collapse=", "), "! \n")

for(param in param_names){
  param_name_raw <- paste0("mu_", param, "_raw")
  
  # Extract posterior samples for each group (probit scale)
  group1_raw <- sims_list[[param_name_raw]][, 1]  # ahn_hc
  group2_raw <- sims_list[[param_name_raw]][, 2]  # gpt_OLD_2023
  group3_raw <- sims_list[[param_name_raw]][, 3]  # gpt_NEW_2025
  
  # Convert to probability scale
  group1_prob <- probit_to_prob(group1_raw)
  group2_prob <- probit_to_prob(group2_raw)
  group3_prob <- probit_to_prob(group3_raw)
  
  # Compute contrasts in probability space (FORGOT TO DO THIS IN DIFF RECOVERY) (cant probit to prob this later since its arithmetic)
  diff_gpt2023_hc <- group2_prob - group1_prob
  diff_gpt2025_hc <- group3_prob - group1_prob
  diff_gpt2025_gpt2023 <- group3_prob - group2_prob
  
  # Store results
  group_contrasts <- rbind(group_contrasts, data.frame(
    parameter = param,
    
    # GPT-2023 vs Human
    diff_gpt2023_hc_median = median(diff_gpt2023_hc),
    diff_gpt2023_hc_q025 = quantile(diff_gpt2023_hc, 0.025),
    diff_gpt2023_hc_q975 = quantile(diff_gpt2023_hc, 0.975),
    prob_gpt2023_greater_hc = mean(diff_gpt2023_hc > 0), #posterior probability of difference being greater than 0
    
    # GPT-2025 vs Human
    diff_gpt2025_hc_median = median(diff_gpt2025_hc),
    diff_gpt2025_hc_q025 = quantile(diff_gpt2025_hc, 0.025),
    diff_gpt2025_hc_q975 = quantile(diff_gpt2025_hc, 0.975),
    prob_gpt2025_greater_hc = mean(diff_gpt2025_hc > 0), #posterior probability of difference being greater than 0
    
    # GPT-2025 vs GPT-2023
    diff_gpt2025_gpt2023_median = median(diff_gpt2025_gpt2023),
    diff_gpt2025_gpt2023_q025 = quantile(diff_gpt2025_gpt2023, 0.025),
    diff_gpt2025_gpt2023_q975 = quantile(diff_gpt2025_gpt2023, 0.975),
    prob_gpt2025_greater_gpt2023 = mean(diff_gpt2025_gpt2023 > 0) #posterior probability of difference being greater than 0
  ))
  
  # Print contrast summaries (quantiles and such)
  cat(sprintf("\n%s:\n", toupper(param)))
  cat(sprintf("  GPT-2023 vs HC: %.3f [%.3f, %.3f] (P(>0)=%.3f)\n",
              median(diff_gpt2023_hc),
              quantile(diff_gpt2023_hc, 0.025),
              quantile(diff_gpt2023_hc, 0.975),
              mean(diff_gpt2023_hc > 0)))
  cat(sprintf("  GPT-2025 vs HC: %.3f [%.3f, %.3f] (P(>0)=%.3f)\n",
              median(diff_gpt2025_hc),
              quantile(diff_gpt2025_hc, 0.025),
              quantile(diff_gpt2025_hc, 0.975),
              mean(diff_gpt2025_hc > 0)))
  cat(sprintf("  GPT-2025 vs GPT-2023: %.3f [%.3f, %.3f] (P(>0)=%.3f)\n",
              median(diff_gpt2025_gpt2023),
              quantile(diff_gpt2025_gpt2023, 0.025),
              quantile(diff_gpt2025_gpt2023, 0.975),
              mean(diff_gpt2025_gpt2023 > 0)))
}

# ---------------------------
# Convergence diagnostics
# ---------------------------
cat("\n=== CONVERGENCE DIAGNOSTICS ===\n")
cat("Max Rhat:", max(summary_mat[, "Rhat"], na.rm = TRUE), "\n")
cat("Min n.eff:", min(summary_mat[, "n.eff"], na.rm = TRUE), "\n")

if(max(summary_mat[, "Rhat"], na.rm = TRUE) > 1.1){
  cat("WARNING: Some Rhat values exceed 1.1, indicating potential convergence issues!\n")
}

# ---------------------------
# Save results
# ---------------------------
write.csv(group_params, file.path(results_dir, "group_parameter_estimates.csv"), row.names = FALSE)
write.csv(group_contrasts, file.path(results_dir, "group_parameter_contrasts.csv"), row.names = FALSE)
saveRDS(samples, file.path(results_dir, "jags_samples.rds"))
saveRDS(sims_list, file.path(results_dir, "posterior_samples.rds"))

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Results saved to:", results_dir, "\n")