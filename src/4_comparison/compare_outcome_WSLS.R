# ============================================================================
# GROUP OUTCOME(!) COMPARISON AFTER VETTING WSLS MODEL
# ============================================================================
# WSLS script to compare the OUTCOME parameters between the real groups (not focused on deck choice) --> using a very simple model
# Adapting the grp recovery script to real data (but only fitting once)
# Using a simpler outcome-only model
# Might report it, might not?

library(pacman)
pacman::p_load(R2jags, parallel, dplyr)
set.seed(1234)

probit_to_prob <- function(x) pnorm(x)
# ---------------------------
# PATHS & LOADING
# ---------------------------
root_path <- "~/IGT_LLM_DecisionMaking"
results_dir <- file.path(root_path, "src/4_comparison/results/outcomes")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
model_file <- file.path(root_path, "models", "simpler_models", "WSLS_hier_grp_compare_outcome.txt") #only focuses on outcome, no deck choice

# Read in data for all groups
old_gpt_data <- read.csv(file.path(root_path, "data", "final_data", "clean_gpt_OLD_2023.csv"))
hc_data <- read.csv(file.path(root_path, "data", "final_data", "clean_ahn_hc.csv"))
new_gpt_data <- read.csv(file.path(root_path, "data", "final_data", "clean_gpt_NEW_2025.csv"))

# Remove probabilities (can't use them anyway)
old_gpt_data <- old_gpt_data %>% select(-probabilities)

# ---------------------------
# DATA PREPARATION
# ---------------------------
n_trials <- 100
group_names <- c("ahn_hc", "gpt_OLD_2023", "gpt_NEW_2025")

# Add group labels
hc_data$group <- "ahn_hc"
old_gpt_data$group <- "gpt_OLD_2023"
new_gpt_data$group <- "gpt_NEW_2025"

# Combine all data
all_data <- rbind(hc_data, old_gpt_data, new_gpt_data)

# Unique subjects per group
subjects_per_group <- all_data %>% 
  group_by(group) %>% 
  summarise(n_subjects = n_distinct(subjID))
print("Subjects per group:")
print(subjects_per_group)

# Create matrices for JAGS
unique_subjects <- unique(all_data$subjID)
n_subjects <- length(unique_subjects)
X_mat <- matrix(NA, nrow = n_subjects, ncol = n_trials)
group_idx <- numeric(n_subjects)

for(i in 1:n_subjects){
  subj_id <- unique_subjects[i]
  subj_data <- all_data[all_data$subjID == subj_id, ]
  
  n_trials_subj <- nrow(subj_data)
  X_mat[i, 1:n_trials_subj] <- subj_data$X  # outcome
  
  # Group index
  group_idx[i] <- which(group_names == subj_data$group[1])
}

# Convert outcomes to binary: win=1, lose=0
X_binary <- ifelse(X_mat >= 0, 1, 0)
#any(is.na(X_binary))

# ---------------------------
# PREPARE JAGS DATA
# ---------------------------
jags_data <- list(
  X = X_binary,
  ntrials = apply(!is.na(X_binary), 1, sum),
  nsubs = n_subjects,
  group = group_idx,
  n_groups = 3
)

# Parameters to monitor (outcome probabilities & variability)
jags_params <- c("mu_prob_win_raw", "lambda_prob_win")  

# ---------------------------
# RUN JAGS
# ---------------------------
cat("\n=== RUNNING HIERARCHICAL OUTCOME COMPARISON ===\n")
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
  n.thin = 2, #to remove autocorrelation when high n.iter, supposedly
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
group_params <- data.frame()
group_contrasts <- data.frame()

# Extract group-level parameters
for(g in 1:length(group_names)){
  group_name <- group_names[g]
  
  mu_prob_win_raw <- sims_list$mu_prob_win_raw[, g]
  lambda_prob_win <- sims_list$lambda_prob_win[, g]
  
  # Convert to probability scale
  mu_prob_win_prob <- probit_to_prob(mu_prob_win_raw)
  
  # Store results
  group_params <- rbind(group_params, data.frame(
    group = group_name,
    mu_prob_win_median = median(mu_prob_win_prob),
    mu_prob_win_q025 = quantile(mu_prob_win_prob, 0.025),
    mu_prob_win_q975 = quantile(mu_prob_win_prob, 0.975),
    rhat_prob_win = summary_mat[paste0("mu_prob_win_raw[", g, "]"), "Rhat"],
    n_eff_prob_win = summary_mat[paste0("mu_prob_win_raw[", g, "]"), "n.eff"]
  ))
  
  cat(sprintf("\n%s: P(win) = %.3f [%.3f, %.3f]\n", 
              toupper(group_name),
              median(mu_prob_win_prob),
              quantile(mu_prob_win_prob, 0.025),
              quantile(mu_prob_win_prob, 0.975)))
}

# ---------------------------
# Compute group contrasts
# ---------------------------
group1_prob <- probit_to_prob(sims_list$mu_prob_win_raw[, 1])
group2_prob <- probit_to_prob(sims_list$mu_prob_win_raw[, 2])
group3_prob <- probit_to_prob(sims_list$mu_prob_win_raw[, 3])

diff_gpt2023_hc <- group2_prob - group1_prob
diff_gpt2025_hc <- group3_prob - group1_prob
diff_gpt2025_gpt2023 <- group3_prob - group2_prob

group_contrasts <- data.frame(
  contrast = c("GPT-2023 vs HC", "GPT-2025 vs HC", "GPT-2025 vs GPT-2023"),
  median = c(median(diff_gpt2023_hc), median(diff_gpt2025_hc), median(diff_gpt2025_gpt2023)),
  q025 = c(quantile(diff_gpt2023_hc, 0.025), quantile(diff_gpt2025_hc, 0.025), quantile(diff_gpt2025_gpt2023, 0.025)),
  q975 = c(quantile(diff_gpt2023_hc, 0.975), quantile(diff_gpt2025_hc, 0.975), quantile(diff_gpt2025_gpt2023, 0.975)),
  prob_greater_0 = c(mean(diff_gpt2023_hc > 0), mean(diff_gpt2025_hc > 0), mean(diff_gpt2025_gpt2023 > 0))
)

cat("\n=== GROUP CONTRASTS ===\n")
print(group_contrasts)

# ---------------------------
# Save results
# ---------------------------
write.csv(group_params, file.path(results_dir, "group_outcome_estimates.csv"), row.names = FALSE)
write.csv(group_contrasts, file.path(results_dir, "group_outcome_contrasts.csv"), row.names = FALSE)
saveRDS(samples, file.path(results_dir, "jags_outcome_samples.rds"))
saveRDS(sims_list, file.path(results_dir, "posterior_outcome_samples.rds"))

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Results saved to:", results_dir, "\n")
