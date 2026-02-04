# ============================================================================
# POSTERIOR PREDICTIVE CHECK FOR WSLS MODEL
# Focus: Switches, Win-Stay/Lose-Switch
# ============================================================================

pacman::p_load(R2jags, ggplot2, dplyr, gridExtra)

# ---- Settings ----
root_path <- "~/IGT_LLM_DecisionMaking"
groups <- c("ahn_hc", "gpt_OLD_2023", "gpt_NEW_2025")
group_name <- groups[3] # which group to do PPC on

# ---- Load data ----
fitfile <- file.path(root_path, "src", "2_estimation", "results", "simpler_models",
                     paste0("jags_fit_WSLS_", group_name, ".RData")) 

#This data is stored in GIT LFS ('git lfs install' to fetch them)
load(fitfile)

data_file <- file.path(root_path, "data", "final_data", paste0("clean_", group_name, ".csv"))
data <- read.csv(data_file)

# ---- Prepare arrays ----
subIDs <- unique(data$subjID)
nsubs <- length(subIDs)
ntrials_max <- 100

x_obs <- array(0, c(nsubs, ntrials_max))
X <- array(0, c(nsubs, ntrials_max))
ntrials <- numeric(nsubs)

for (s in 1:nsubs) {
  ntrials[s] <- sum(data$subjID == subIDs[s])
  x_obs[s,] <- data$x[data$subjID == subIDs[s]]
  X[s,] <- data$X[data$subjID == subIDs[s]]
}
X <- X / 100

# Extract predictions
x_pred <- samples$BUGSoutput$sims.list$x_pred
n_sims <- dim(x_pred)[1]

plot_dir <- file.path(root_path, "src", "2_estimation",
                      "posterior_predictive_checks", "posterior_predictive_plots")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================================
# STATISTIC 1: SWITCHES
# ============================================================================

calc_switches <- function(choices, ntrials) {
  switches <- numeric(nsubs)
  for (s in 1:nsubs) {
    switches[s] <- sum(choices[s, 1:(ntrials[s]-1)] != choices[s, 2:ntrials[s]])
  }
  return(switches)
}

obs_switches <- calc_switches(x_obs, ntrials)
pred_switches <- t(apply(x_pred, 1, calc_switches, ntrials = ntrials))

# Bayesian p-value
obs_total <- sum(obs_switches)
pred_total <- rowSums(pred_switches)
pval_switches <- mean(pred_total >= obs_total)

# Plot
p1 <- ggplot() +
  geom_histogram(aes(x = pred_total), fill = "skyblue", alpha = 0.6, bins = 50) +
  geom_vline(xintercept = obs_total, color = "red", size = 1.5, linetype = "dashed") +
  annotate("text", x = obs_total, y = Inf, 
           label = sprintf("Obs = %d\np = %.3f", obs_total, pval_switches), 
           vjust = 1.5, hjust = -0.1, color = "red", fontface = "bold", size = 5) +
  labs(title = paste("Total Switches (WSLS Model):", group_name),
       x = "Total Switches", y = "Count") +
  theme_bw(base_size = 14)

ggsave(file.path(plot_dir, paste0("switches_WSLS_", group_name, ".png")),
       p1, width = 8, height = 6)

# ============================================================================
# STATISTIC 2: WIN-STAY / LOSE-SWITCH PROBABILITIES
# ============================================================================

calc_wsls <- function(choices, outcomes, ntrials) {
  # outcomes: X matrix (payoffs/100)
  # Returns matrix: [nsubs x 4] with win-stay, win-switch, lose-stay, lose-switch probs
  
  wsls <- matrix(NA, nrow = nsubs, ncol = 4)
  colnames(wsls) <- c("win_stay", "win_switch", "lose_stay", "lose_switch")
  
  for (s in 1:nsubs) {
    n <- ntrials[s]
    if (n < 2) next
    
    win_stay <- 0
    win_switch <- 0
    lose_stay <- 0
    lose_switch <- 0
    
    for (t in 1:(n-1)) {
      current_choice <- choices[s, t]
      next_choice <- choices[s, t+1]
      outcome <- outcomes[s, t]
      
      stayed <- (current_choice == next_choice)
      won <- (outcome > 0)  # simple win = positive outcome
      
      if (won) {
        if (stayed) {
          win_stay <- win_stay + 1
        } else {
          win_switch <- win_switch + 1
        }
      } else {  # lost
        if (stayed) {
          lose_stay <- lose_stay + 1
        } else {
          lose_switch <- lose_switch + 1
        }
      }
    }
    
    # Calculate probabilities
    total_wins <- win_stay + win_switch
    total_losses <- lose_stay + lose_switch
    
    if (total_wins > 0) {
      wsls[s, "win_stay"] <- win_stay / total_wins
      wsls[s, "win_switch"] <- win_switch / total_wins
    }
    if (total_losses > 0) {
      wsls[s, "lose_stay"] <- lose_stay / total_losses
      wsls[s, "lose_switch"] <- lose_switch / total_losses
    }
  }
  
  return(wsls)
}

# Calculate for observed data
obs_wsls <- calc_wsls(x_obs, X, ntrials)

# Calculate for predicted data
pred_wsls <- array(NA, dim = c(n_sims, nsubs, 4))
for (i in 1:n_sims) {
  pred_wsls[i,,] <- calc_wsls(x_pred[i,,], X, ntrials)
}

# win stay lose switch, prepare data
wsls_plot_data <- data.frame(
  Strategy = rep(c("Win-Stay", "Lose-Switch"), each = nsubs * 2),
  Probability = c(
    obs_wsls[, 1],  # win_stay
    colMeans(pred_wsls[,, 1], na.rm = TRUE),  # win_stay
    obs_wsls[, 4],  # lose_switch
    colMeans(pred_wsls[,, 4], na.rm = TRUE)  # lose_switch
  ),
  Type = rep(rep(c("Observed", "Predicted"), each = nsubs), 2)
)

# Remove NAs
wsls_plot_data <- wsls_plot_data[!is.na(wsls_plot_data$Probability), ]

p4 <- ggplot(wsls_plot_data, aes(x = Strategy, y = Probability, fill = Type)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.3) +
  scale_fill_manual(values = c("Observed" = "coral", "Predicted" = "skyblue")) +
  ylim(0, 1) +
  labs(title = paste("Win-Stay / Lose-Switch (WSLS Model):", group_name),
       y = "Probability", x = "") +
  theme_bw(base_size = 14) +
  theme(legend.position = "top")

ggsave(file.path(plot_dir, paste0("wsls_WSLS_", group_name, ".png")),
       p4, width = 8, height = 6)

# ============================================================================
# SUMMARY REPORT
# ============================================================================

summary_file <- file.path(root_path, "src", "2_estimation",
                          "posterior_predictive_checks", "posterior_predictive_plots", 
                          paste0("PPC_summary_WSLS_", group_name, ".txt"))

sink(summary_file)

cat("\n===============================================\n")
cat("POSTERIOR PREDICTIVE CHECK SUMMARY\n")
cat("===============================================\n")
cat("Group:", group_name, "\n")
cat("Model: WSLS (Win-Stay-Lose-Shift)\n\n")
cat("IF p value is <0.05 or >0.95, there is a significant mismatch between model and data!\n\n")

cat("SWITCHES:\n")
cat(sprintf("  Observed total: %d\n", obs_total))
cat(sprintf("  Predicted mean: %.1f [%.1f, %.1f]\n", 
            mean(pred_total), 
            quantile(pred_total, 0.025),
            quantile(pred_total, 0.975)))
cat(sprintf("  Bayesian p-value: %.3f", pval_switches))

if (pval_switches < 0.05 || pval_switches > 0.95) {
  cat(" *")
}
cat("\n")

cat("\nWIN-STAY / LOSE-SWITCH:\n")
obs_ws_mean <- mean(obs_wsls[, "win_stay"], na.rm = TRUE)
obs_ls_mean <- mean(obs_wsls[, "lose_switch"], na.rm = TRUE)

pred_ws_means <- rowMeans(pred_wsls[,, 1], na.rm = TRUE)
pred_ls_means <- rowMeans(pred_wsls[,, 4], na.rm = TRUE)

pval_ws <- mean(pred_ws_means >= obs_ws_mean)
pval_ls <- mean(pred_ls_means >= obs_ls_mean)

cat(sprintf("  Win-Stay:    Obs=%.3f, Pred=%.3f [%.3f, %.3f], p=%.3f", 
            obs_ws_mean,
            mean(pred_ws_means),
            quantile(pred_ws_means, 0.025),
            quantile(pred_ws_means, 0.975),
            pval_ws))
if (pval_ws < 0.05 || pval_ws > 0.95) {
  cat(" *")
}
cat("\n")

cat(sprintf("  Lose-Switch: Obs=%.3f, Pred=%.3f [%.3f, %.3f], p=%.3f",
            obs_ls_mean,
            mean(pred_ls_means),
            quantile(pred_ls_means, 0.025),
            quantile(pred_ls_means, 0.975),
            pval_ls))
if (pval_ls < 0.05 || pval_ls > 0.95) {
  cat(" *")
}
cat("\n")

sink()
cat("Summary saved to:", summary_file, "\n")