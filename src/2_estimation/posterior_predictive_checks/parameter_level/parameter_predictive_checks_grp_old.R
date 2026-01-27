
# Posterior Predictive Simulation for All Parameters (FOR THE OLD MODEL) (PARAMETER LEVEL)
# Simulates synthetic subjects from hierarchical posterior draws
# Saves simulated subject-level parameters to CSV
#THIS ISNT A FULL POSTERIOR PREDICTIVE CHECK
#old model constrains the values?

# Load required packages
pacman::p_load(ggplot2, dplyr, tidyr)
set.seed(2026)

# -------------------------------
# Paths and group names
# -------------------------------
root_path <- "C:/Users/Igiba/Documents/IGT_LLM_DecisionMaking"
groups <- c("ahn_hc", "gpt_OLD_2023", "gpt_NEW_2025")  # human → old → new

# -------------------------------
# Load group-level posterior draws
# -------------------------------
param_list <- list()
for(g in groups){
  param_file <- file.path(root_path, "src", "2_estimation", "results", "old_model",
                          paste0("param_estimated_fixedTheta_", g, ".csv"))
  param_list[[g]] <- read.csv(param_file)
}

# -------------------------------
# Function to simulate synthetic subjects (OLD model)
# -------------------------------
simulate_subjects_model <- function(params, nsub=500){
  # Sample one posterior draw
  row <- params[sample(1:nrow(params), 1), ]
  
  # Convert precisions to SDs
  sigma_a_rew <- 1 / sqrt(max(row$lambda_a_rew, 1e-6))
  sigma_a_pun <- 1 / sqrt(max(row$lambda_a_pun, 1e-6))
  sigma_K     <- 1 / sqrt(max(row$lambda_K, 1e-6))
  sigma_omega_f <- 1 / sqrt(max(row$lambda_omega_f, 1e-6))
  sigma_omega_p <- 1 / sqrt(max(row$lambda_omega_p, 1e-6))
  
  # Simulate subject-level parameters
  a_rew <- rnorm(nsub, mean=row$mu_a_rew, sd=sigma_a_rew)
  a_rew <- pmax(a_rew, 0)  # truncated at 0
  
  a_pun <- rnorm(nsub, mean=row$mu_a_pun, sd=sigma_a_pun)
  a_pun <- pmax(a_pun, 0)  # truncated at 0
  
  K <- rnorm(nsub, mean=row$mu_K, sd=sigma_K)
  K <- pmax(K, 0)  # truncated at 0
  
  omega_f <- rnorm(nsub, mean=row$mu_omega_f, sd=sigma_omega_f)
  omega_p <- rnorm(nsub, mean=row$mu_omega_p, sd=sigma_omega_p)
  
  df <- data.frame(a_rew, a_pun, K, omega_f, omega_p)
  return(df)
}

# -------------------------------
# Simulate synthetic subjects per group (average across multiple draws)
# -------------------------------
n_draws <- 50  # number of posterior draws to average over
nsub <- 500    # subjects per draw

sim_list <- list()
sd_list <- list()

for(g in groups){
  param_df <- param_list[[g]]
  
  # Store all simulated subjects across draws
  sim_all <- list()
  sd_matrix <- matrix(NA, nrow=n_draws, ncol=5)
  colnames(sd_matrix) <- c("a_rew","a_pun","K","omega_f","omega_p")
  
  for(i in 1:n_draws){
    sim <- simulate_subjects_model(param_df, nsub=nsub)
    sim_all[[i]] <- sim
    sd_matrix[i,] <- apply(sim, 2, sd)
  }
  
  # Combine all draws for posterior predictive plot
  sim_list[[g]] <- bind_rows(sim_all)
  
  # Store mean SD and SD of SDs for error bars
  sd_list[[g]] <- data.frame(
    parameter = colnames(sd_matrix),
    mean_SD = apply(sd_matrix, 2, mean),
    SD_of_SD = apply(sd_matrix, 2, sd)
  )
}

# -------------------------------
# Combine all groups into one long data frame
# -------------------------------
sim_df_all <- bind_rows(
  lapply(groups, function(g){
    df <- sim_list[[g]]
    df$group <- g
    df
  })
)

# -------------------------------
# Save simulated subject-level parameters to CSV
# -------------------------------
save_file <- file.path(root_path,  "src", "2_estimation", "posterior_predictive_checks", "parameter_level", "simulated_subjects_all_parameters_OLD_model.csv")
if(!dir.exists(dirname(save_file))) dir.create(dirname(save_file), recursive = TRUE)
write.csv(sim_df_all, save_file, row.names=FALSE)
cat("Simulated subject-level parameters saved to:\n", save_file, "\n")

# -------------------------------
# Plot posterior predictive densities for all parameters
# -------------------------------
sim_df_long <- sim_df_all %>%
  pivot_longer(cols = c(a_rew, a_pun, K, omega_f, omega_p),
               names_to = "parameter", values_to = "value")

plot_all <- ggplot(sim_df_long, aes(x=value, color=group)) +
  geom_density(size=1) +
  facet_wrap(~parameter, scales="free") +
  theme_minimal() +
  labs(title="Parameter posterior predictive distributions across groups (OLD MODEL)",
       x="Parameter value", y="Density") +
  scale_color_manual(values=c("ahn_hc"="black", "gpt_OLD_2023"="blue", "gpt_NEW_2025"="red")) +
  theme(text=element_text(size=14), legend.title=element_blank())

plot_file <- file.path(root_path, "src", "2_estimation", "posterior_predictive_checks", "parameter_level", "posterior_predictive_all_parameters_OLD_model.png")
ggsave(plot_file, plot_all, width=12, height=8, dpi=400)
cat("Density plot saved to:\n", plot_file, "\n")

# -------------------------------
# Visualize subject-level SDs with error bars
# -------------------------------
sd_table_all <- bind_rows(
  lapply(names(sd_list), function(g){
    df <- sd_list[[g]]
    df$group <- g
    df
  })
)

# Order factors
sd_table_all$parameter <- factor(sd_table_all$parameter,
                                 levels = c("a_rew","a_pun","K","omega_f","omega_p"))
sd_table_all$group <- factor(sd_table_all$group, levels=c("ahn_hc","gpt_OLD_2023","gpt_NEW_2025"))

# Plot
sd_plot <- ggplot(sd_table_all, aes(x=parameter, y=mean_SD, fill=group)) +
  geom_bar(stat="identity", position=position_dodge(width=0.8), color="black") +
  geom_errorbar(aes(ymin=mean_SD - SD_of_SD, ymax=mean_SD + SD_of_SD),
                width=0.2, position=position_dodge(width=0.8)) +
  scale_fill_manual(values=c("ahn_hc"="black", "gpt_OLD_2023"="blue", "gpt_NEW_2025"="red")) +
  theme_minimal() +
  labs(title="Subject-level SDs for IGT parameters (OLD model)",
       x="Parameter", y="Mean SD ± SD of SDs") +
  theme(text=element_text(size=14), legend.title=element_blank())

plot_file_sd <- file.path(root_path, "src", "2_estimation", "posterior_predictive_checks", "parameter_level", "subject_level_SD_plot_OLD_model.png")
ggsave(plot_file_sd, sd_plot, width=10, height=6, dpi=400)

sd_plot