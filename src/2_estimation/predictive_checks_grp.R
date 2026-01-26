# Hierarchical ORL model posterior predictive check script (group-level)

#install.packages("pacman")
pacman::p_load(R2jags, parallel, ggplot2, tidyr)

# set seed
set.seed(1234)

# set data grps
groups <- c("ahn_hc", "gpt_EN", "gpt_JP")

### SECTION WHERE YOU CAN CHANGE THINGS ###
fixed_theta <- TRUE
model_tag <- ifelse(fixed_theta, "fixedTheta", "freeTheta")

# new root path
root_path <- "~/IGT_LLM_DecisionMaking"

# select group
group_name = groups[1] # grp1: human, grp2: gpt_en, grp3: gpt_jp

# number of simulated datasets to generate
n_sims <- 100

### CODE ###
# load original data
file <- file.path(root_path, "data", "final_data", paste0("clean_", group_name, ".csv"))
data <- read.csv(file)

# extract relevant variables
subIDs <- unique(data$subjID)
nsubs <- length(subIDs)
x_raw <- data$x
X_raw <- data$X

# initialize empty arrays
ntrials_max <- 100
x <- array(0, c(nsubs, ntrials_max))
X <- array(0, c(nsubs, ntrials_max))
ntrials <- array(0, c(nsubs))

# turn data from long format into arrays
print(paste0("Working on file: ", basename(file)))
for (s in 1:nsubs) {
  ntrials[s] <- length(x_raw[data$subjID==subIDs[s]])
  X_sub <- X_raw[data$subjID==subIDs[s]] 
  x_sub <- x_raw[data$subjID==subIDs[s]] 
  x[s,] <- x_sub
  X[s,] <- X_sub
}

# scale the payoffs
X <- X/100

# load posterior samples from hierarchical model
posterior_file <- file.path(root_path, "src", "2_estimation", "results",
                            paste0("grouplevel_", model_tag, "_", group_name, ".csv"))
posterior_samples <- read.csv(posterior_file)

# set timer
start_time = Sys.time()
print(paste0(start_time, " - Starting group-level posterior predictive checks"))

# initialize arrays to store simulated data
sim_choices <- array(0, c(n_sims, nsubs, ntrials_max))
sim_deck_props <- array(0, c(n_sims, nsubs, 4)) # proportion of choices per deck
sim_learning_curves <- array(0, c(n_sims, nsubs, 5)) # good deck choices in 5 blocks

# function to simulate choices from ORL model
simulate_ORL <- function(a_rew, a_pun, K, omega_f, omega_p, X_all, ntrials, theta=3) {
  
  # initialize values
  ev <- rep(0, 4)
  ef <- rep(0, 4)
  ps <- rep(0, 4)
  p <- rep(0, 4)
  choices <- rep(0, ntrials)
  
  for (t in 1:ntrials) {
    
    # compute choice probabilities
    for (d in 1:4) {
      p[d] <- exp(theta * (ev[d] + ef[d] + ps[d]))
    }
    p <- p / sum(p)
    
    # make choice
    choices[t] <- sample(1:4, 1, prob=p)
    x_t <- choices[t]
    
    # get outcome
    X_t <- X_all[t]
    
    # update values
    ev_pu <- ev
    ef_pu <- ef
    
    if (X_t >= 0) {
      # reward trial
      ev[x_t] <- ev[x_t] + a_rew * (X_t - ev[x_t])
      ef[x_t] <- ef[x_t] + a_rew * (X_t - ef[x_t])
    } else {
      # punishment trial
      ev[x_t] <- ev[x_t] + a_pun * (X_t - ev[x_t])
      ef[x_t] <- ef[x_t] + a_pun * (X_t - ef[x_t])
    }
    
    # update perseveration
    ps <- ps * K
    ps[x_t] <- ps[x_t] + omega_f * abs(ef[x_t] - ef_pu[x_t]) + omega_p * abs(ev[x_t] - ev_pu[x_t])
  }
  
  return(choices)
}

# run simulations
print(paste0("Running ", n_sims, " simulations..."))
for (sim in 1:n_sims) {
  
  # sample one draw from posterior
  sample_idx <- sample(1:nrow(posterior_samples), 1)
  
  # extract group-level parameters for this draw
  mu_a_rew <- posterior_samples$mu_a_rew[sample_idx]
  mu_a_pun <- posterior_samples$mu_a_pun[sample_idx]
  mu_K <- posterior_samples$mu_K[sample_idx]
  mu_omega_f <- posterior_samples$mu_omega_f[sample_idx]
  mu_omega_p <- posterior_samples$mu_omega_p[sample_idx]
  
  sigma_a_rew <- posterior_samples$sigma_a_rew[sample_idx]
  sigma_a_pun <- posterior_samples$sigma_a_pun[sample_idx]
  sigma_K <- posterior_samples$sigma_K[sample_idx]
  sigma_omega_f <- posterior_samples$sigma_omega_f[sample_idx]
  sigma_omega_p <- posterior_samples$sigma_omega_p[sample_idx]
  
  # simulate parameters for each subject from group distribution
  for (s in 1:nsubs) {
    
    # sample individual parameters from group distribution
    a_rew <- rnorm(1, mu_a_rew, sigma_a_rew)
    a_pun <- rnorm(1, mu_a_pun, sigma_a_pun)
    K <- rnorm(1, mu_K, sigma_K)
    omega_f <- rnorm(1, mu_omega_f, sigma_omega_f)
    omega_p <- rnorm(1, mu_omega_p, sigma_omega_p)
    
    # apply transformations to keep in valid range
    a_rew <- plogis(a_rew) # bound to [0,1]
    a_pun <- plogis(a_pun)
    K <- plogis(K)
    omega_f <- exp(omega_f) # positive only
    omega_p <- exp(omega_p)
    
    # simulate choices
    ntrials_s <- ntrials[s]
    sim_choices[sim, s, 1:ntrials_s] <- simulate_ORL(a_rew, a_pun, K, omega_f, omega_p, 
                                                     X[s, 1:ntrials_s], ntrials_s)
    
    # calculate deck proportions
    for (deck in 1:4) {
      sim_deck_props[sim, s, deck] <- sum(sim_choices[sim, s, 1:ntrials_s] == deck) / ntrials_s
    }
    
    # calculate learning curve (proportion good decks per block)
    block_size <- 20
    for (block in 1:5) {
      block_start <- (block - 1) * block_size + 1
      block_end <- block * block_size
      block_choices <- sim_choices[sim, s, block_start:block_end]
      sim_learning_curves[sim, s, block] <- sum(block_choices %in% c(3, 4)) / block_size
    }
  }
  
  if (sim %% 10 == 0) {
    print(paste0("Completed ", sim, "/", n_sims, " simulations"))
  }
}

# calculate observed statistics for comparison
obs_deck_props <- array(0, c(nsubs, 4))
obs_learning_curves <- array(0, c(nsubs, 5))

for (s in 1:nsubs) {
  ntrials_s <- ntrials[s]
  
  # observed deck proportions
  for (deck in 1:4) {
    obs_deck_props[s, deck] <- sum(x[s, 1:ntrials_s] == deck) / ntrials_s
  }
  
  # observed learning curve
  block_size <- 20
  for (block in 1:5) {
    block_start <- (block - 1) * block_size + 1
    block_end <- block * block_size
    block_choices <- x[s, block_start:block_end]
    obs_learning_curves[s, block] <- sum(block_choices %in% c(3, 4)) / block_size
  }
}

# summarize simulated vs observed deck proportions
cat("\n=== Deck Selection Proportions ===\n")
for (deck in 1:4) {
  obs_mean <- mean(obs_deck_props[, deck])
  sim_mean <- mean(apply(sim_deck_props[, , deck], 1, mean))
  sim_lower <- quantile(apply(sim_deck_props[, , deck], 1, mean), 0.025)
  sim_upper <- quantile(apply(sim_deck_props[, , deck], 1, mean), 0.975)
  
  cat(sprintf("Deck %d - Observed: %.3f | Simulated: %.3f [%.3f, %.3f]\n",
              deck, obs_mean, sim_mean, sim_lower, sim_upper))
}

# summarize learning curves
cat("\n=== Learning Curves (Good Deck Proportion) ===\n")
for (block in 1:5) {
  obs_mean <- mean(obs_learning_curves[, block])
  sim_mean <- mean(apply(sim_learning_curves[, , block], 1, mean))
  sim_lower <- quantile(apply(sim_learning_curves[, , block], 1, mean), 0.025)
  sim_upper <- quantile(apply(sim_learning_curves[, , block], 1, mean), 0.975)
  
  cat(sprintf("Block %d - Observed: %.3f | Simulated: %.3f [%.3f, %.3f]\n",
              block, obs_mean, sim_mean, sim_lower, sim_upper))
}

# create visualization: deck proportions
deck_data <- data.frame(
  Deck = rep(1:4, each = nsubs),
  Observed = as.vector(t(obs_deck_props)),
  Subject = rep(1:nsubs, 4)
)

sim_deck_summary <- data.frame(
  Deck = 1:4,
  Mean = apply(apply(sim_deck_props, c(1, 3), mean), 2, mean),
  Lower = apply(apply(sim_deck_props, c(1, 3), mean), 2, quantile, 0.025),
  Upper = apply(apply(sim_deck_props, c(1, 3), mean), 2, quantile, 0.975)
)

plot_deck <- file.path(root_path, "src", "2_estimation", "plots", "ppc",
                       paste0("ppc_deck_props_", model_tag, "_", group_name, ".png"))

png(plot_deck, width = 2000, height = 1400, res = 300)
p1 <- ggplot() +
  geom_point(data = deck_data, aes(x = Deck, y = Observed), 
             alpha = 0.3, position = position_jitter(width = 0.1)) +
  geom_point(data = sim_deck_summary, aes(x = Deck, y = Mean), 
             color = "red", size = 3) +
  geom_errorbar(data = sim_deck_summary, aes(x = Deck, ymin = Lower, ymax = Upper),
                color = "red", width = 0.2) +
  labs(title = paste0("Posterior Predictive Check: Deck Proportions (", group_name, ")"),
       subtitle = "Black points = observed subjects | Red = simulated mean ± 95% CI",
       y = "Proportion of Choices") +
  theme_minimal() +
  theme(text = element_text(size = 12))
print(p1)
dev.off()

# create visualization: learning curves
learning_data <- data.frame(
  Block = rep(1:5, each = nsubs),
  Observed = as.vector(t(obs_learning_curves)),
  Subject = rep(1:nsubs, 5)
)

sim_learning_summary <- data.frame(
  Block = 1:5,
  Mean = apply(apply(sim_learning_curves, c(1, 3), mean), 2, mean),
  Lower = apply(apply(sim_learning_curves, c(1, 3), mean), 2, quantile, 0.025),
  Upper = apply(apply(sim_learning_curves, c(1, 3), mean), 2, quantile, 0.975)
)

plot_learning <- file.path(root_path, "src", "2_estimation", "plots", "ppc",
                           paste0("ppc_learning_curves_", model_tag, "_", group_name, ".png"))

png(plot_learning, width = 2000, height = 1400, res = 300)
p2 <- ggplot() +
  geom_line(data = learning_data, aes(x = Block, y = Observed, group = Subject),
            alpha = 0.2) +
  geom_line(data = sim_learning_summary, aes(x = Block, y = Mean),
            color = "red", size = 1.5) +
  geom_ribbon(data = sim_learning_summary, aes(x = Block, ymin = Lower, ymax = Upper),
              fill = "red", alpha = 0.2) +
  labs(title = paste0("Posterior Predictive Check: Learning Curves (", group_name, ")"),
       subtitle = "Gray lines = observed subjects | Red = simulated mean ± 95% CI",
       y = "Proportion Good Decks (C + D)",
       x = "Block (20 trials each)") +
  theme_minimal() +
  theme(text = element_text(size = 12))
print(p2)
dev.off()

# save summary statistics
ppc_summary <- data.frame(
  group = group_name,
  model = model_tag,
  n_sims = n_sims,
  deck1_obs = mean(obs_deck_props[, 1]),
  deck1_sim = mean(apply(sim_deck_props[, , 1], 1, mean)),
  deck2_obs = mean(obs_deck_props[, 2]),
  deck2_sim = mean(apply(sim_deck_props[, , 2], 1, mean)),
  deck3_obs = mean(obs_deck_props[, 3]),
  deck3_sim = mean(apply(sim_deck_props[, , 3], 1, mean)),
  deck4_obs = mean(obs_deck_props[, 4]),
  deck4_sim = mean(apply(sim_deck_props[, , 4], 1, mean))
)

savefile <- file.path(root_path, "src", "2_estimation", "results",
                      paste0("ppc_summary_", model_tag, "_", group_name, ".csv"))
write.csv(ppc_summary, savefile, row.names = FALSE)

# print total time
end_time = Sys.time()
run_time = round(end_time - start_time, 2)
print(paste0("Total run time: ", run_time))

cat("\n=== Posterior Predictive Check Complete ===\n")
cat("Plots saved to: src/2_estimation/plots/ppc/\n")
cat("Summary saved to: src/2_estimation/results/\n")