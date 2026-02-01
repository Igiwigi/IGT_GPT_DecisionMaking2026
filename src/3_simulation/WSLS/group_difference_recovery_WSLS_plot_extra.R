# ============================================================================
# POST-HOC VISUALIZATION EDITOR FOR GROUP DIFFERENCE RECOVERY
# ============================================================================
# This script loads the saved results from the parameter recovery simulation & makes plots
# gotta make the plots better for reporting & prettier (this has been sloppily thrown together)
# ============================================================================

library(pacman)
pacman::p_load(ggplot2, dplyr, tidyr, gridExtra, cowplot)

# ---------------------------
# CONFIGURATION
# ---------------------------
root_path <- "~/IGT_LLM_DecisionMaking"
results_dir <- file.path(root_path, "src/3_simulation/WSLS/recovery/results/group_diff")
output_dir <- file.path(results_dir, "revised_plots")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Plot settings
PLOT_WIDTH <- 2400
PLOT_HEIGHT <- 800
PLOT_DPI <- 300
BASE_FONT_SIZE <- 14

# Color schemes
GROUP_COLORS <- list(
  human = "#3498db",      # Blue
  gpt2023 = "#e74c3c",    # Red
  gpt2025 = "#2ecc71"     # Green
)

CONTRAST_COLORS <- c(
  "#e67e22",  # Orange - GPT2023 vs Human
  "#9b59b6",  # Purple - GPT2025 vs Human
  "#34495e"   # Dark gray - GPT2025 vs GPT2023
)

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

# Convert probit to probability
probit_to_prob <- function(x) {
  pnorm(x)
}

# Calculate recovery metrics
calc_metrics <- function(true_vals, rec_vals) {
  list(
    bias = mean(rec_vals - true_vals, na.rm = TRUE),
    rmse = sqrt(mean((rec_vals - true_vals)^2, na.rm = TRUE)),
    mae = mean(abs(rec_vals - true_vals), na.rm = TRUE),
    r = cor(true_vals, rec_vals, use = "complete.obs"),
    r_squared = cor(true_vals, rec_vals, use = "complete.obs")^2
  )
}

# Format parameter names for display
format_param_name <- function(param) {
  param_labels <- c(
    "prob_stay_win" = "P(Stay|Win)",
    "prob_stay_lose" = "P(Stay|Lose)",
    "epsilon" = "Exploration (ε)"
  )
  return(param_labels[param])
}

# ---------------------------
# LOAD DATA
# ---------------------------
cat("Loading results...\n")

# Check if files exist
required_files <- c(
  "group_params_recovery.csv",
  "group_contrasts_recovery.csv",
  "group_hyperparams.csv"
)

for (file in required_files) {
  if (!file.exists(file.path(results_dir, file))) {
    stop(sprintf("Required file not found: %s\nPlease run the main recovery simulation first.", file))
  }
}

results <- read.csv(file.path(results_dir, "group_params_recovery.csv"))
contrasts <- read.csv(file.path(results_dir, "group_contrasts_recovery.csv"))
hyperparams <- read.csv(file.path(results_dir, "group_hyperparams.csv"))

n_simulations <- max(results$sim_id)
cat(sprintf("Loaded results from %d simulations\n", n_simulations))

# ---------------------------
# PRINT SUMMARY STATISTICS
# ---------------------------
cat("\n=== RECOVERY PERFORMANCE SUMMARY ===\n")

for (group_name in c("human", "gpt2023", "gpt2025")) {
  cat(sprintf("\n%s:\n", toupper(group_name)))
  res_g <- results[results$group == group_name, ]
  
  for (param in c("prob_stay_win", "prob_stay_lose", "epsilon")) {
    true_vals_probit <- res_g[[paste0("true_mu_", param)]]
    rec_vals_probit <- res_g[[paste0("rec_mu_", param)]]
    
    true_vals <- probit_to_prob(true_vals_probit)
    rec_vals <- probit_to_prob(rec_vals_probit)
    
    metrics <- calc_metrics(true_vals, rec_vals)
    
    cat(sprintf("  %s:\n", format_param_name(param)))
    cat(sprintf("    True value: %.3f\n", unique(true_vals)[1]))
    cat(sprintf("    Mean recovered: %.3f (SD=%.3f)\n", mean(rec_vals), sd(rec_vals)))
    cat(sprintf("    Bias: %.3f | RMSE: %.3f | r: %.3f\n", 
                metrics$bias, metrics$rmse, metrics$r))
  }
}

# ============================================================================
# PLOT 1: INDIVIDUAL GROUP PARAMETER RECOVERY (IMPROVED)
# ============================================================================
cat("\n=== Creating individual group parameter recovery plots ===\n")

for (group_name in c("human", "gpt2023", "gpt2025")) {
  res_g <- results[results$group == group_name, ]
  
  png(file.path(output_dir, sprintf("recovery_%s_params.png", group_name)), 
      width = PLOT_WIDTH, height = PLOT_HEIGHT, res = PLOT_DPI)
  
  par(mfrow = c(1, 3), mar = c(5, 5, 4, 2), oma = c(0, 0, 3, 0), bg = "white")
  
  group_color <- GROUP_COLORS[[group_name]]
  
  for (param in c("prob_stay_win", "prob_stay_lose", "epsilon")) {
    true_vals <- probit_to_prob(res_g[[paste0("true_mu_", param)]])
    rec_vals <- probit_to_prob(res_g[[paste0("rec_mu_", param)]])
    
    metrics <- calc_metrics(true_vals, rec_vals)
    
    # Determine axis limits with some padding
    all_vals <- c(true_vals, rec_vals)
    x_range <- range(true_vals)
    y_range <- range(rec_vals)
    plot_min <- max(0, min(all_vals) - 0.05)
    plot_max <- min(1, max(all_vals) + 0.05)
    
    # Create plot
    plot(true_vals, rec_vals,
         main = format_param_name(param),
         xlab = "True Probability", 
         ylab = "Recovered Probability",
         xlim = c(plot_min, plot_max), 
         ylim = c(plot_min, plot_max),
         pch = 21, 
         bg = adjustcolor(group_color, alpha = 0.6), 
         col = group_color, 
         cex = 2, 
         lwd = 1.5,
         cex.lab = 1.3, 
         cex.axis = 1.2, 
         cex.main = 1.4,
         las = 1, 
         bty = "l")
    
    # Add grid
    grid(col = "gray85", lty = 1)
    
    # Re-plot points on top
    points(true_vals, rec_vals,
           pch = 21, 
           bg = adjustcolor(group_color, alpha = 0.6), 
           col = group_color, 
           cex = 2, 
           lwd = 1.5)
    
    # Identity line
    abline(a = 0, b = 1, col = "black", lwd = 2.5, lty = 2)
    
    # True value reference line
    true_val <- unique(true_vals)[1]
    abline(h = true_val, col = group_color, lty = 3, lwd = 2)
    abline(v = true_val, col = group_color, lty = 3, lwd = 2)
    
    # Add metrics text box
    legend("bottomright", 
           legend = c(
             sprintf("True: %.3f", true_val),
             sprintf("Mean: %.3f", mean(rec_vals)),
             sprintf("Bias: %+.3f", metrics$bias),
             sprintf("RMSE: %.3f", metrics$rmse),
             sprintf("r: %.3f", metrics$r)
           ),
           bty = "n", 
           cex = 1.1, 
           text.col = "gray20",
           bg = "white")
  }
  
  mtext(sprintf("Parameter Recovery: %s", toupper(group_name)), 
        outer = TRUE, cex = 1.6, font = 2)
  
  dev.off()
}

# ============================================================================
# PLOT 2: COMBINED PARAMETER RECOVERY (ALL GROUPS)
# ============================================================================
cat("\n=== Creating combined parameter recovery plot ===\n")

png(file.path(output_dir, "recovery_all_groups_combined.png"),
    width = PLOT_WIDTH, height = PLOT_HEIGHT, res = PLOT_DPI)

par(mfrow = c(1, 3), mar = c(5, 5, 4, 2), oma = c(0, 0, 3, 0), bg = "white")

for (param in c("prob_stay_win", "prob_stay_lose", "epsilon")) {
  # Get data for all groups
  true_vals_all <- probit_to_prob(results[[paste0("true_mu_", param)]])
  rec_vals_all <- probit_to_prob(results[[paste0("rec_mu_", param)]])
  groups <- results$group
  
  # Calculate overall metrics
  metrics <- calc_metrics(true_vals_all, rec_vals_all)
  
  # Determine axis limits
  all_vals <- c(true_vals_all, rec_vals_all)
  plot_min <- max(0, min(all_vals) - 0.05)
  plot_max <- min(1, max(all_vals) + 0.05)
  
  # Create empty plot
  plot(true_vals_all, rec_vals_all, type = "n",
       main = format_param_name(param),
       xlab = "True Probability", 
       ylab = "Recovered Probability",
       xlim = c(plot_min, plot_max), 
       ylim = c(plot_min, plot_max),
       cex.lab = 1.3, 
       cex.axis = 1.2, 
       cex.main = 1.4,
       las = 1, 
       bty = "l")
  
  # Add grid
  grid(col = "gray85", lty = 1)
  
  # Plot each group
  for (grp in c("human", "gpt2023", "gpt2025")) {
    idx <- groups == grp
    points(true_vals_all[idx], rec_vals_all[idx], 
           pch = 21, 
           bg = adjustcolor(GROUP_COLORS[[grp]], alpha = 0.5), 
           col = GROUP_COLORS[[grp]], 
           cex = 1.8, 
           lwd = 1.5)
  }
  
  # Identity line
  abline(a = 0, b = 1, col = "black", lwd = 2.5, lty = 2)
  
  # Legend
  legend("topleft", 
         legend = c(
           sprintf("Overall r: %.3f", metrics$r),
           sprintf("RMSE: %.3f", metrics$rmse),
           "",
           "Human", "GPT-2023", "GPT-2025"
         ),
         pch = c(NA, NA, NA, rep(21, 3)),
         pt.bg = c(NA, NA, NA, unlist(GROUP_COLORS)),
         col = c(NA, NA, NA, unlist(GROUP_COLORS)),
         pt.cex = 1.8,
         bty = "n", 
         cex = 1.0)
}

mtext("Parameter Recovery Across All Groups", 
      outer = TRUE, cex = 1.6, font = 2)

dev.off()

# ============================================================================
# PLOT 3: LAMBDA (PRECISION) RECOVERY
# ============================================================================
cat("\n=== Creating lambda (precision) recovery plot ===\n")

png(file.path(output_dir, "recovery_lambda_precision.png"),
    width = PLOT_WIDTH, height = PLOT_HEIGHT, res = PLOT_DPI)

par(mfrow = c(1, 3), mar = c(5, 5, 4, 2), oma = c(0, 0, 3, 0), bg = "white")

for (param in c("prob_stay_win", "prob_stay_lose", "epsilon")) {
  true_vals <- results[[paste0("true_lambda_", param)]]
  rec_vals <- results[[paste0("rec_lambda_", param)]]
  groups <- results$group
  
  metrics <- calc_metrics(true_vals, rec_vals)
  
  # Create empty plot
  plot(true_vals, rec_vals, type = "n",
       main = format_param_name(param),
       xlab = "True Precision (λ)", 
       ylab = "Recovered Precision (λ)",
       cex.lab = 1.3, 
       cex.axis = 1.2, 
       cex.main = 1.4,
       las = 1, 
       bty = "l")
  
  grid(col = "gray85", lty = 1)
  
  # Plot by group
  for (grp in c("human", "gpt2023", "gpt2025")) {
    idx <- groups == grp
    points(true_vals[idx], rec_vals[idx], 
           pch = 21, 
           bg = adjustcolor(GROUP_COLORS[[grp]], alpha = 0.5), 
           col = GROUP_COLORS[[grp]], 
           cex = 1.8, 
           lwd = 1.5)
  }
  
  # Identity line
  abline(a = 0, b = 1, col = "black", lwd = 2.5, lty = 2)
  
  # Legend
  legend("topleft", 
         legend = c(
           sprintf("r: %.3f", metrics$r),
           sprintf("RMSE: %.3f", metrics$rmse),
           sprintf("Bias: %+.3f", metrics$bias),
           "",
           "Human", "GPT-2023", "GPT-2025"
         ),
         pch = c(NA, NA, NA, NA, rep(21, 3)),
         pt.bg = c(NA, NA, NA, NA, unlist(GROUP_COLORS)),
         col = c(NA, NA, NA, NA, unlist(GROUP_COLORS)),
         pt.cex = 1.8,
         bty = "n", 
         cex = 1.0)
}

mtext("Precision (Lambda) Recovery Across All Groups", 
      outer = TRUE, cex = 1.6, font = 2)

dev.off()

# ============================================================================
# PLOT 4: GROUP CONTRASTS (IMPROVED)
# ============================================================================
cat("\n=== Creating group contrasts plot ===\n")

png(file.path(output_dir, "recovery_group_contrasts.png"),
    width = PLOT_WIDTH, height = PLOT_HEIGHT, res = PLOT_DPI)

par(mfrow = c(1, 3), mar = c(5, 5, 4, 2), oma = c(0, 0, 3, 0), bg = "white")

contrast_names <- c("GPT-2023 vs Human", "GPT-2025 vs Human", "GPT-2025 vs GPT-2023")

for (param in c("prob_stay_win", "prob_stay_lose", "epsilon")) {
  con_p <- contrasts[contrasts$parameter == param, ]
  
  # Collect all contrasts
  all_true <- c()
  all_rec <- c()
  contrast_groups <- c()
  
  for (i in 1:nrow(con_p)) {
    for (contrast_idx in 1:3) {
      if (contrast_idx == 1) {
        true_diff <- con_p$true_diff_gpt2023_human[i]
        rec_diff <- con_p$rec_diff_gpt2023_human[i]
      } else if (contrast_idx == 2) {
        true_diff <- con_p$true_diff_gpt2025_human[i]
        rec_diff <- con_p$rec_diff_gpt2025_human[i]
      } else {
        true_diff <- con_p$true_diff_gpt2025_gpt2023[i]
        rec_diff <- con_p$rec_diff_gpt2025_gpt2023[i]
      }
      
      all_true <- c(all_true, true_diff)
      all_rec <- c(all_rec, rec_diff)
      contrast_groups <- c(contrast_groups, contrast_idx)
    }
  }
  
  metrics <- calc_metrics(all_true, all_rec)
  
  # Create plot
  plot(all_true, all_rec,
       main = format_param_name(param),
       xlab = "True Difference (probit scale)", 
       ylab = "Recovered Difference (probit scale)",
       pch = 21, 
       bg = adjustcolor(CONTRAST_COLORS[contrast_groups], alpha = 0.6), 
       col = CONTRAST_COLORS[contrast_groups],
       cex = 1.8, 
       lwd = 1.5,
       cex.lab = 1.3, 
       cex.axis = 1.2, 
       cex.main = 1.4,
       las = 1, 
       bty = "l")
  
  grid(col = "gray85", lty = 1)
  
  # Re-plot points
  points(all_true, all_rec,
         pch = 21, 
         bg = adjustcolor(CONTRAST_COLORS[contrast_groups], alpha = 0.6), 
         col = CONTRAST_COLORS[contrast_groups],
         cex = 1.8, 
         lwd = 1.5)
  
  # Reference lines
  abline(a = 0, b = 1, col = "black", lwd = 2.5, lty = 2)
  abline(h = 0, v = 0, col = "gray40", lty = 3, lwd = 1.5)
  
  # Legend
  legend("topleft", 
         legend = c(
           sprintf("r: %.3f", metrics$r),
           sprintf("RMSE: %.3f", metrics$rmse),
           "",
           contrast_names
         ),
         pch = c(NA, NA, NA, rep(21, 3)),
         pt.bg = c(NA, NA, NA, CONTRAST_COLORS),
         col = c(NA, NA, NA, CONTRAST_COLORS),
         pt.cex = 1.8,
         bty = "n", 
         cex = 1.0)
}

mtext("Group Difference Recovery", 
      outer = TRUE, cex = 1.6, font = 2)

dev.off()

# ============================================================================
# PLOT 5: RECOVERY PLOT SCATTER
# ============================================================================
# Prepare data for ggplot (get true, recovered, and them as prob)
plot_data <- results %>%
  select(sim_id, group, starts_with("true_mu_"), starts_with("rec_mu_")) %>%
  pivot_longer(
    cols = -c(sim_id, group),
    names_to = "variable",
    values_to = "value"
  ) %>%
  mutate(
    parameter = case_when(
      grepl("prob_stay_win", variable) ~ "prob_stay_win",
      grepl("prob_stay_lose", variable) ~ "prob_stay_lose",
      grepl("epsilon", variable) ~ "epsilon"
    ),
    type = ifelse(grepl("^true_", variable), "true", "recovered"),
    value_prob = probit_to_prob(value)
  ) %>%
  select(-variable) %>%
  pivot_wider(
    names_from = type,
    values_from = c(value, value_prob)
  )

# Parameter recovery scatter plot
p1 <- ggplot(plot_data, aes(x = value_prob_true, y = value_prob_recovered, color = group)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", size = 1, color = "black") +
  geom_point(size = 3, alpha = 0.6) +
  facet_wrap(~parameter, scales = "free", 
             labeller = labeller(parameter = c(
               prob_stay_win = "P(Stay|Win)",
               prob_stay_lose = "P(Stay|Lose)",
               epsilon = "Exploration (ε)"
             ))) +
  scale_color_manual(
    values = unlist(GROUP_COLORS),
    labels = c("Human", "GPT-2023", "GPT-2025")
  ) +
  labs(
    title = "Parameter Recovery Across All Groups",
    x = "True Probability",
    y = "Recovered Probability",
    color = "Group"
  ) +
  theme_minimal(base_size = BASE_FONT_SIZE) +
  theme(
    plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
    strip.text = element_text(face = "bold", size = 14),
    strip.background = element_rect(fill = "gray95", color = NA),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_line(color = "gray95"),
    panel.border = element_rect(color = "gray80", fill = NA)
  )

ggsave(file.path(output_dir, "recovery_ggplot_scatter.png"),
       plot = p1, width = 12, height = 5, dpi = PLOT_DPI, bg = "white")

# ============================================================================
# PLOT 6: DISTRIBUTION COMPARISON
# ============================================================================
cat("\n=== Creating distribution comparison plots ===\n")

# Prepare data for distributions
dist_data <- results %>%
  select(sim_id, group, starts_with("rec_mu_")) %>%
  pivot_longer(
    cols = starts_with("rec_mu_"),
    names_to = "parameter",
    values_to = "value"
  ) %>%
  mutate(
    parameter = gsub("rec_mu_", "", parameter),
    value_prob = probit_to_prob(value)
  )

# Add true values for reference
true_vals <- hyperparams %>%
  select(group, starts_with("mu_")) %>%
  group_by(group) %>%
  summarise(
    prob_stay_win = mean(mu_prob_stay_win),
    prob_stay_lose = mean(mu_prob_stay_lose),
    epsilon = mean(mu_epsilon)
  ) %>%
  pivot_longer(
    cols = -group,
    names_to = "parameter",
    values_to = "true_value"
  )

dist_data <- dist_data %>%
  left_join(true_vals, by = c("group", "parameter"))

p2 <- ggplot(dist_data, aes(x = value_prob, fill = group, color = group)) +
  geom_density(alpha = 0.3, size = 1) +
  geom_vline(aes(xintercept = true_value, color = group), 
             linetype = "dashed", size = 1) +
  facet_wrap(~parameter, scales = "free",
             labeller = labeller(parameter = c(
               prob_stay_win = "P(Stay|Win)",
               prob_stay_lose = "P(Stay|Lose)",
               epsilon = "Exploration (ε)"
             ))) +
  scale_fill_manual(
    values = unlist(GROUP_COLORS),
    labels = c("Human", "GPT-2023", "GPT-2025")
  ) +
  scale_color_manual(
    values = unlist(GROUP_COLORS),
    labels = c("Human", "GPT-2023", "GPT-2025")
  ) +
  labs(
    title = sprintf("Posterior Distributions Across %d Simulations", n_simulations),
    subtitle = "Dashed lines show true parameter values",
    x = "Probability",
    y = "Density",
    fill = "Group",
    color = "Group"
  ) +
  theme_minimal(base_size = BASE_FONT_SIZE) +
  theme(
    plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray40"),
    strip.text = element_text(face = "bold", size = 14),
    strip.background = element_rect(fill = "gray95", color = NA),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_line(color = "gray95"),
    panel.border = element_rect(color = "gray80", fill = NA)
  )

ggsave(file.path(output_dir, "recovery_distributions.png"),
       plot = p2, width = 12, height = 5, dpi = PLOT_DPI, bg = "white")

# ============================================================================
# PLOT 7: BIAS AND PRECISION COMPARISON
# ============================================================================
cat("\n=== Creating bias and precision comparison plot ===\n")

# Calculate bias and SD for each group and parameter
summary_stats <- plot_data %>%
  group_by(group, parameter) %>%
  summarise(
    bias = mean(value_prob_recovered - value_prob_true, na.rm = TRUE),
    sd = sd(value_prob_recovered, na.rm = TRUE),
    rmse = sqrt(mean((value_prob_recovered - value_prob_true)^2, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(
    parameter_label = case_when(
      parameter == "prob_stay_win" ~ "P(Stay|Win)",
      parameter == "prob_stay_lose" ~ "P(Stay|Lose)",
      parameter == "epsilon" ~ "Exploration (ε)"
    )
  )

# Bias plot
p3a <- ggplot(summary_stats, aes(x = parameter_label, y = bias, fill = group)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  scale_fill_manual(
    values = unlist(GROUP_COLORS),
    labels = c("Human", "GPT-2023", "GPT-2025")
  ) +
  labs(
    title = "Recovery Bias by Group and Parameter",
    x = "Parameter",
    y = "Bias (Recovered - True)",
    fill = "Group"
  ) +
  theme_minimal(base_size = BASE_FONT_SIZE) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    legend.position = "top",
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )

# RMSE plot
p3b <- ggplot(summary_stats, aes(x = parameter_label, y = rmse, fill = group)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(
    values = unlist(GROUP_COLORS),
    labels = c("Human", "GPT-2023", "GPT-2025")
  ) +
  labs(
    title = "Recovery RMSE by Group and Parameter",
    x = "Parameter",
    y = "RMSE",
    fill = "Group"
  ) +
  theme_minimal(base_size = BASE_FONT_SIZE) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    legend.position = "top",
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )

p3 <- plot_grid(p3a, p3b, ncol = 2)

ggsave(file.path(output_dir, "recovery_bias_rmse_comparison.png"),
       plot = p3, width = 14, height = 6, dpi = PLOT_DPI, bg = "white")

# ============================================================================
# SAVE SUMMARY STATISTICS
# ============================================================================
cat("\n=== Saving summary statistics ===\n")

write.csv(summary_stats, 
          file.path(output_dir, "recovery_summary_stats.csv"), 
          row.names = FALSE)

# Create a comprehensive summary table
comprehensive_summary <- plot_data %>%
  group_by(group, parameter) %>%
  summarise(
    true_prob = mean(value_prob_true, na.rm = TRUE),
    mean_recovered = mean(value_prob_recovered, na.rm = TRUE),
    sd_recovered = sd(value_prob_recovered, na.rm = TRUE),
    bias = mean(value_prob_recovered - value_prob_true, na.rm = TRUE),
    rmse = sqrt(mean((value_prob_recovered - value_prob_true)^2, na.rm = TRUE)),
    mae = mean(abs(value_prob_recovered - value_prob_true), na.rm = TRUE),
    r = cor(value_prob_true, value_prob_recovered, use = "complete.obs"),
    .groups = "drop"
  ) %>%
  arrange(group, parameter)

write.csv(comprehensive_summary, 
          file.path(output_dir, "comprehensive_recovery_summary.csv"), 
          row.names = FALSE)

print(comprehensive_summary)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
cat("\n" %>% paste0(rep("=", 80), collapse = "") %>% paste0("\n"))
cat("VISUALIZATION COMPLETE!\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

cat("Output directory:", output_dir, "\n\n")

cat("Files created:\n")
cat("  1. recovery_human_params.png\n")
cat("  2. recovery_gpt2023_params.png\n")
cat("  3. recovery_gpt2025_params.png\n")
cat("  4. recovery_all_groups_combined.png\n")
cat("  5. recovery_lambda_precision.png\n")
cat("  6. recovery_group_contrasts.png\n")
cat("  7. recovery_ggplot_scatter.png\n")
cat("  8. recovery_distributions.png\n")
cat("  9. recovery_bias_rmse_comparison.png\n")
cat(" 10. recovery_summary_stats.csv\n")
cat(" 11. comprehensive_recovery_summary.csv\n\n")

cat("Summary statistics:\n")
cat(sprintf("  Total simulations: %d\n", n_simulations))
cat(sprintf("  Groups analyzed: %s\n", paste(unique(results$group), collapse = ", ")))
cat(sprintf("  Parameters: %s\n", paste(c("prob_stay_win", "prob_stay_lose", "epsilon"), collapse = ", ")))

cat("\n" %>% paste0(rep("=", 80), collapse = "") %>% paste0("\n"))