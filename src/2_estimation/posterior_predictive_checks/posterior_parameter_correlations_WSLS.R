# -------------------------------
# Posterior parameter correlation / identifiability plots
# -------------------------------

if(!require(pacman)) install.packages("pacman")
pacman::p_load(ggplot2, bayesplot)

#note: only one chain, should be fine for now, its not even reported

# -------------------------------
# Paths and groups
# -------------------------------
root_path <- "~/IGT_LLM_DecisionMaking"
root_path <- "C:/Users/Igiba/Documents/decision_making_course/IGT_GPT_DecisionMaking2026" #for laptop

groups <- c("ahn_hc", "gpt_OLD_2023", "gpt_NEW_2025")
param_path <- file.path(root_path, "src/2_estimation/results/simpler_models")
cor_dir <- file.path(root_path, "src/2_estimation/posterior_predictive_checks/parameter_correlations")

if(!dir.exists(cor_dir)) dir.create(cor_dir, recursive = TRUE)

bayesplot_theme_set(theme_bw())

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
  
  #not in this version of the model file, so have to get it here
  param_list[[g]]$sigma_prob_stay_win    <- 1 / sqrt(param_list[[g]]$lambda_prob_stay_win)
  param_list[[g]]$sigma_prob_switch_lose <- 1 / sqrt(param_list[[g]]$lambda_prob_switch_lose)
  
  #sample a random subset
  idx <- sample(nrow(param_list[[g]]), min(1000, nrow(param_list[[g]])))
  posterior_sample <- param_list[[g]][idx, ]
  
  # Select WSLS parameters of interest
  param_cols <- c("mu_prob_stay_win", "mu_prob_switch_lose",
                  "sigma_prob_stay_win", "sigma_prob_switch_lose")
  
  # Check which columns actually exist in the data
  available_cols <- param_cols[param_cols %in% colnames(posterior_sample)]
  
  #check if missing
  if(length(available_cols) < length(param_cols)){
    stop("Missing parameter columns for group ", g, ": ",
         paste(setdiff(param_cols, available_cols), collapse = ", "),
         ". Available columns: ", paste(colnames(posterior_sample), collapse = ", "))
  }
  
  posterior_mat <- as.matrix(posterior_sample[, available_cols])

  
  # Create more readable column names for plotting
  colnames(posterior_mat) <- gsub("mu_prob_stay_win", "μ(Stay|Win)", colnames(posterior_mat))
  colnames(posterior_mat) <- gsub("mu_prob_switch_lose", "μ(Switch|Lose)", colnames(posterior_mat))
  colnames(posterior_mat) <- gsub("sigma_prob_stay_win", "σ(Stay|Win)", colnames(posterior_mat))
  colnames(posterior_mat) <- gsub("sigma_prob_switch_lose", "σ(Switch|Lose)", colnames(posterior_mat))

  # Create an array with dimensions (iterations, chains, parameters) for compatibility with bayesplot
  posterior_array <- array(posterior_mat,
                           dim = c(nrow(posterior_mat), 1, ncol(posterior_mat)))
  dimnames(posterior_array) <- list(iterations = NULL,
                                    chains = "chain:1",
                                    parameters = colnames(posterior_mat))
  
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

