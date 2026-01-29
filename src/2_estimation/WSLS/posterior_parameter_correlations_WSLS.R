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
param_path <- file.path(root_path, "src/2_estimation/results/simpler_models")
cor_dir <- file.path(root_path, "src/2_estimation/WSLS/posterior_predictive_checks/parameter_correlations")

if(!dir.exists(cor_dir)) dir.create(cor_dir, recursive = TRUE)

# -------------------------------
# Load posterior draws for each group
# -------------------------------
param_list <- lapply(groups, function(g){
  file <- file.path(param_path, paste0("param_estimated_WSLS_", g, ".csv"))
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
  
  # Select WSLS parameters of interest
  param_cols <- c("mu_prob_stay_win", "mu_prob_stay_lose", "mu_epsilon",
                  "sigma_prob_stay_win", "sigma_prob_stay_lose", "sigma_epsilon")
  
  # Check which columns actually exist in the data
  available_cols <- param_cols[param_cols %in% colnames(posterior_sample)]
  
  if(length(available_cols) == 0){
    warning("No matching parameter columns found for group ", g, 
            ". Available columns: ", paste(colnames(posterior_sample), collapse=", "))
    next
  }
  
  posterior_mat <- as.matrix(posterior_sample[, available_cols])
  
  posterior_array <- array(posterior_mat,
                           dim = c(nrow(posterior_mat), 1, ncol(posterior_mat)))
  
  dimnames(posterior_array) <- list(NULL, "chain1", colnames(posterior_mat))
  
  # Create more readable column names for plotting
  colnames(posterior_mat) <- gsub("mu_prob_stay_win", "μ(Stay|Win)", colnames(posterior_mat))
  colnames(posterior_mat) <- gsub("mu_prob_stay_lose", "μ(Stay|Lose)", colnames(posterior_mat))
  colnames(posterior_mat) <- gsub("mu_epsilon", "μ(ε)", colnames(posterior_mat))
  colnames(posterior_mat) <- gsub("sigma_prob_stay_win", "σ(Stay|Win)", colnames(posterior_mat))
  colnames(posterior_mat) <- gsub("sigma_prob_stay_lose", "σ(Stay|Lose)", colnames(posterior_mat))
  colnames(posterior_mat) <- gsub("sigma_epsilon", "σ(ε)", colnames(posterior_mat))
  
  dimnames(posterior_array)[[3]] <- colnames(posterior_mat)
  
  # Generate the pairs plot
  pairs_plot <- mcmc_pairs(
    posterior_array,
    diag_fun = "hist",
    off_diag_fun = "scatter",
    off_diag_args = list(alpha = 0.3, size = 0.5)
    )
  
  # Save plot
  ggsave(
    filename = file.path(cor_dir, paste0("posterior_pairs_WSLS_", g, ".png")),
    plot = pairs_plot,
    width = 12, height = 10, dpi = 300
  )
  
  cat("  Saved to:", file.path(cor_dir, paste0("posterior_pairs_WSLS_", g, ".png")), "\n")
}

cat("\nAll pairs plots generated successfully!\n")
