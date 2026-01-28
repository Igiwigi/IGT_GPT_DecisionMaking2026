# -------------------------------
# Posterior parameter correlation / identifiability plots
# -------------------------------

if(!require(pacman)) install.packages("pacman")
pacman::p_load(ggplot2, bayesplot)

# -------------------------------
# Paths and groups
# -------------------------------
root_path <- "~/IGT_LLM_DecisionMaking"
groups <- c("ahn_hc", "gpt_OLD_2023", "gpt_NEW_2025")
param_path <- file.path(root_path, "src/2_estimation/results/new_model")
cor_dir <- file.path(root_path, "src/2_estimation/posterior_predictive_checks/parameter_correlations")

if(!dir.exists(cor_dir)) dir.create(cor_dir, recursive = TRUE)

# -------------------------------
# Load posterior draws for each group
# -------------------------------
param_list <- lapply(groups, function(g){
  file <- file.path(param_path, paste0("param_estimated_fixedTheta_", g, ".csv"))
  if(!file.exists(file)) stop("Posterior file not found: ", file)
  read.csv(file)
})
names(param_list) <- groups

# -------------------------------
# Generate and save pairs plots
# -------------------------------
for(g in groups){
  cat("Posterior pairs plot for group:", g, "\n")
  
  posterior_sample <- param_list[[g]][1:min(1000, nrow(param_list[[g]])), ]  # subset for clarity
  
  # Select only parameters of interest
  param_cols <- c("mu_a_rew", "mu_a_pun", "mu_K", "mu_omega_f", "mu_omega_p",
                  "sigma_a_rew", "sigma_a_pun", "sigma_K", "sigma_omega_f", "sigma_omega_p")
  
  posterior_mat <- as.matrix(posterior_sample[, param_cols])
  
  # Generate the pairs plot
  pairs_plot <- mcmc_pairs(
    posterior_mat,
    diag_fun = "hist",
    off_diag_fun = "scatter",
    off_diag_args = list(alpha = 0.3, size = 0.5)
  )
  
  # Save plot
  ggsave(
    filename = file.path(cor_dir, paste0("posterior_pairs_", g, ".png")),
    plot = pairs_plot,
    width = 12, height = 10, dpi = 300
  )
}
