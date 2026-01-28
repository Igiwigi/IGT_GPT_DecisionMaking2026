# ============================================================================
# SIMPLIFIED POSTERIOR PREDICTIVE CHECK
# Focus: Switches, Deck Proportions, Learning Trend
# ============================================================================

pacman::p_load(R2jags, ggplot2, dplyr, gridExtra)

# ---- Settings ----
root_path <- "~/IGT_LLM_DecisionMaking"
groups <- c("ahn_hc", "gpt_OLD_2023", "gpt_NEW_2025")
group_name <- groups[3]
fixed_theta <- TRUE
model_tag <- "fixedTheta"

# ---- Load data ----
fitfile <- file.path(root_path, "src", "2_estimation", "results", "new_model",
                     paste0("jags_fit_", model_tag, "_", group_name, ".RData"))
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

plot_dir <- file.path(root_path, "src", "2_estimation", 'posterior_predictive_checks', "posterior_predictive_plots")
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
  geom_vline(xintercept = obs_total, color = "red", linewidth = 1.5, linetype = "dashed") +
  annotate("text", x = obs_total, y = Inf, 
           label = sprintf("Obs = %d\np = %.3f", obs_total, pval_switches), 
           vjust = 1.5, hjust = -0.1, color = "red", fontface = "bold", linewidth = 5) +
  labs(title = paste("Total Switches:", group_name),
       x = "Total Switches", y = "Count") +
  theme_minimal(base_size = 14)

ggsave(file.path(plot_dir, paste0("switches_", group_name, ".png")),
       p1, width = 8, height = 6)

# ============================================================================
# STATISTIC 2: DECK PROPORTIONS
# ============================================================================

calc_deck_props <- function(choices, ntrials) {
  props <- matrix(NA, nrow = nsubs, ncol = 4)
  for (s in 1:nsubs) {
    for (deck in 1:4) {
      props[s, deck] <- sum(choices[s, 1:ntrials[s]] == deck) / ntrials[s]
    }
  }
  return(props)
}

obs_deck_props <- calc_deck_props(x_obs, ntrials)
pred_deck_props <- array(NA, dim = c(n_sims, nsubs, 4))
for (i in 1:n_sims) {
  pred_deck_props[i,,] <- calc_deck_props(x_pred[i,,], ntrials)
}

# Prepare plot data
plot_data <- data.frame()
for (deck in 1:4) {
  obs_df <- data.frame(deck = LETTERS[deck],
                       proportion = obs_deck_props[,deck],
                       type = "Observed")
  pred_mean <- colMeans(pred_deck_props[,,deck])
  pred_df <- data.frame(deck = LETTERS[deck],
                        proportion = pred_mean,
                        type = "Predicted")
  plot_data <- rbind(plot_data, obs_df, pred_df)
}

p2 <- ggplot(plot_data, aes(x = deck, y = proportion, fill = type)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("Observed" = "coral", "Predicted" = "skyblue")) +
  labs(title = paste("Deck Proportions:", group_name),
       y = "Proportion", x = "Deck") +
  theme_minimal(base_size = 14)

ggsave(file.path(plot_dir, paste0("decks_", group_name, ".png")),
       p2, width = 8, height = 6)

# ============================================================================
# STATISTIC 3: LEARNING TREND
# ============================================================================

calc_good_deck_trend <- function(choices, ntrials, n_blocks = 5) {
  trends <- matrix(NA, nrow = nsubs, ncol = n_blocks)
  for (s in 1:nsubs) {
    block_size <- ntrials[s] / n_blocks
    for (b in 1:n_blocks) {
      start_idx <- floor((b-1) * block_size) + 1
      end_idx <- floor(b * block_size)
      trends[s, b] <- mean(choices[s, start_idx:end_idx] %in% c(3, 4))
    }
  }
  return(trends)
}

obs_trends <- calc_good_deck_trend(x_obs, ntrials)
pred_trends <- array(NA, dim = c(n_sims, nsubs, 5))
for (i in 1:n_sims) {
  pred_trends[i,,] <- calc_good_deck_trend(x_pred[i,,], ntrials)
}

# Aggregate
trend_summary <- data.frame()
for (block in 1:5) {
  trend_summary <- rbind(trend_summary,
                         data.frame(block = block,
                                    mean = mean(obs_trends[,block]),
                                    type = "Observed"),
                         data.frame(block = block,
                                    mean = mean(pred_trends[,,block]),
                                    type = "Predicted")
  )
}

p3 <- ggplot(trend_summary, aes(x = block, y = mean, color = type, group = type)) +
  geom_line(size = 1.5) +
  geom_point(size = 3) +
  scale_color_manual(values = c("Observed" = "coral", "Predicted" = "skyblue")) +
  labs(title = paste("Learning Curve:", group_name),
       x = "Block", y = "Proportion Good Decks (C & D)") +
  theme_minimal(base_size = 14)

ggsave(file.path(plot_dir, paste0("learning_", group_name, ".png")),
       p3, width = 8, height = 6)

# ============================================================================
# SUMMARY REPORT
# ============================================================================

cat("\n===============================================\n")
cat("POSTERIOR PREDICTIVE CHECK SUMMARY\n")
cat("===============================================\n")
cat("Group:", group_name, "\n")
cat("Model: Fixed theta = 1\n\n")

cat("SWITCHES:\n")
cat(sprintf("  Observed total: %d\n", obs_total))
cat(sprintf("  Predicted mean: %.1f [%.1f, %.1f]\n", 
            mean(pred_total), 
            quantile(pred_total, 0.025),
            quantile(pred_total, 0.975)))
cat(sprintf("  Bayesian p-value: %.3f\n", pval_switches))

if (pval_switches < 0.05) {
  cat("  >> Model predicts TOO MANY switches\n")
  cat("  >> Subjects are MORE DETERMINISTIC than theta=1\n")
} else if (pval_switches > 0.95) {
  cat("  >> Model predicts TOO FEW switches\n")
  cat("  >> Subjects are MORE EXPLORATORY than theta=1\n")
} else {
  cat("  >> Good fit!\n")
}

cat("\nDECK PROPORTIONS:\n")
for (deck in 1:4) {
  cat(sprintf("  Deck %s: Obs=%.2f, Pred=%.2f\n", 
              LETTERS[deck],
              mean(obs_deck_props[,deck]),
              mean(pred_deck_props[,,deck])))
}

cat("\nLEARNING:\n")
cat(sprintf("  Final block good decks: Obs=%.2f, Pred=%.2f\n",
            mean(obs_trends[,5]),
            mean(pred_trends[,,5])))

cat("\n===============================================\n\n")