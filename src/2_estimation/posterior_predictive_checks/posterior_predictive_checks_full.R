# -------------------------------
# Full Posterior Predictive Check for All Groups (USING NEW ORL MODEL)
# Simulates IGT choices using posterior parameters,
# compares simulated vs real behavior (net scores
# and learning curves), and saves plots.
# -------------------------------

#this still doesnt fully work, might need to fix some things
#the plots look wrong still (and might want different plots? learning? deck choice?)

if(!require(pacman)) install.packages("pacman")
pacman::p_load(ggplot2, dplyr, tidyr)
set.seed(2026)

# -------------------------------
# Paths and groups
# -------------------------------
root_path <- "~/IGT_LLM_DecisionMaking"
groups <- c("ahn_hc", "gpt_OLD_2023", "gpt_NEW_2025")

param_path <- file.path(root_path, "src/2_estimation/results/new_model")
data_path <- file.path(root_path, "data", "final_data")
payoff_file <- file.path(root_path, "utils", "payoff_scheme_3.csv")
ppc_dir <- file.path(root_path, "src/2_estimation/posterior_predictive_checks")

# -------------------------------
# Load payoff scheme
# -------------------------------
payoffs <- read.csv(payoff_file)

payoffs_matrix <- data.frame(
  A = payoffs$win_A - payoffs$loss_A,
  B = payoffs$win_B - payoffs$loss_B,
  C = payoffs$win_C - payoffs$loss_C,
  D = payoffs$win_D - payoffs$loss_D
) # 60 rows, 4 decks

ntrials <- 100  # total trials to simulate
ndeck <- ncol(payoffs_matrix)
n_payoff <- nrow(payoffs_matrix)

# -------------------------------
# Load real behavioral data for each group (and make sure it fits the rest of the code)
# -------------------------------
real_data_list <- lapply(seq_along(groups), function(i){
  g <- groups[i]
  df <- read.csv(file.path(data_path, paste0("clean_", g, ".csv")))
  
  # Rename subject column (to 'subject' from either subjID or ID)
  if("subjID" %in% colnames(df)) df <- dplyr::rename(df, subject = subjID)
  if("ID" %in% colnames(df)) df <- dplyr::rename(df, subject = ID)
  
  # Ensure integer so it doesnt crash out later
  df$subject <- as.integer(df$subject)
  df$group <- g  #need groups for later
  df <- dplyr::rename(df, deck = x)
  df$deck <- as.integer(df$deck)  #need deck for later (from x)
  
  df <- df %>%
    arrange(subject) %>%
    group_by(subject) %>%
    mutate(trial = row_number()) %>%  #add trials after sorting them
    ungroup()
  
  df
})
names(real_data_list) <- groups
colnames(real_data_list[[1]]) #to check they're right

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
# Check posterior correlations / identifiability (and save them)
# -------------------------------
cor_dir <- file.path(ppc_dir, "parameter_correlations")
if(!dir.exists(cor_dir)) dir.create(cor_dir, recursive = TRUE)

if(!require(bayesplot)) install.packages("bayesplot")
library(bayesplot)

for(g in groups){
  cat("Posterior pairs plot for group:", g, "\n")
  
  # Take first 1000 draws (or all if small) for clarity
  posterior_sample <- param_list[[g]][1:min(1000, nrow(param_list[[g]])), ]
  
  # Select only parameters of interest
  param_cols <- c("mu_a_rew", "mu_a_pun", "mu_K", "mu_omega_f", "mu_omega_p",
                  "sigma_a_rew", "sigma_a_pun", "sigma_K", "sigma_omega_f", "sigma_omega_p")
  
  posterior_mat <- as.matrix(posterior_sample[, param_cols])
  
  # Generate the pairs plot
  pairs_plot <- bayesplot::mcmc_pairs(posterior_mat,
                                      diag_fun = "hist",
                                      off_diag_fun = "scatter",
                                      off_diag_args = list(alpha = 0.3, size=0.5))
  
  # Save using ggsave
  ggsave(filename = file.path(cor_dir, paste0("posterior_pairs_", g, ".png")),
         plot = pairs_plot,
         width = 12, height = 10, dpi = 300)
}

# -------------------------------
# Sample subject-level parameters (ORL model, non-centered)
# -------------------------------
sample_subject_params <- function(group_params, nsub){
  posterior_sample <- group_params[sample(1:nrow(group_params), 1), ]
  
  # Extract hyperparameters
  mu_a_rew <- posterior_sample$mu_a_rew
  mu_a_pun <- posterior_sample$mu_a_pun
  mu_K <- posterior_sample$mu_K
  mu_omega_f <- posterior_sample$mu_omega_f
  mu_omega_p <- posterior_sample$mu_omega_p
  sigma_a_rew <- posterior_sample$sigma_a_rew
  sigma_a_pun <- posterior_sample$sigma_a_pun
  sigma_K <- posterior_sample$sigma_K
  sigma_omega_f <- posterior_sample$sigma_omega_f
  sigma_omega_p <- posterior_sample$sigma_omega_p
  
  # Non-centered z's
  z_a_rew <- rnorm(nsub, 0, 1)
  z_a_pun <- rnorm(nsub, 0, 1)
  z_K     <- rnorm(nsub, 0, 1)
  z_of    <- rnorm(nsub, 0, 1)
  z_op    <- rnorm(nsub, 0, 1)
  
  inv_logit <- function(x) 1 / (1 + exp(-x))
  
  subject_params <- data.frame(
    a_rew = inv_logit(mu_a_rew + sigma_a_rew * z_a_rew),
    a_pun = inv_logit(mu_a_pun + sigma_a_pun * z_a_pun),
    K     = exp(mu_K + sigma_K * z_K),
    omega_f = mu_omega_f + sigma_omega_f * z_of,
    omega_p = mu_omega_p + sigma_omega_p * z_op,
    theta = 1
  )
  
  return(subject_params)
}

# -------------------------------
# Simulate trial-level choices (deck depletion handled)
# -------------------------------
simulate_igt <- function(subject_params, ntrials, payoffs_matrix, max_per_deck = 60){
  
  nsub <- nrow(subject_params)
  ndeck <- ncol(payoffs_matrix)
  sim_data <- list()
  
  for(s in 1:nsub){
    
    # Subject parameters
    a_rew <- subject_params$a_rew[s]
    a_pun <- subject_params$a_pun[s]
    K     <- subject_params$K[s]
    omega_f <- subject_params$omega_f[s]
    omega_p <- subject_params$omega_p[s]
    theta <- 1
    
    # Initialize trial-level variables
    Ev_prev <- rep(0, ndeck)
    Ef_prev <- rep(0, ndeck)
    PS_prev <- rep(1, ndeck)
    
    x <- numeric(ntrials)
    X <- numeric(ntrials)
    
    # Track how many times each deck has been chosen
    deck_counts <- rep(0, ndeck)
    
    for(t in 1:ntrials){
      # Compute V and choice probabilities
      V <- Ev_prev + Ef_prev*omega_f + PS_prev*omega_p
      V[is.na(V)] <- 0
      V <- pmin(pmax(V, -700), 700) #so it doesnt overflow; limiting it to a range of -700 and 700
      exp_p <- exp(theta*V)
      if(all(exp_p == 0) || any(is.na(exp_p))) exp_p <- rep(1, ndeck)
      
      # --- Deck depletion ---
      available_decks <- which(deck_counts < max_per_deck)
      if(length(available_decks) == 0){
        x[t] <- NA
        X[t] <- 0
      } else {
        exp_p_avail <- exp_p[available_decks]
        p_avail <- exp_p_avail / sum(exp_p_avail)
        x[t] <- sample(available_decks, 1, prob=p_avail)
        
        # Next unused payoff
        row_index <- deck_counts[x[t]] + 1
        X[t] <- as.numeric(payoffs_matrix[row_index, x[t]])
        deck_counts[x[t]] <- deck_counts[x[t]] + 1
      }
      
      # Update Ev, Ef, PS if deck chosen
      if(!is.na(x[t])){
        # Ev update
        Ev_curr <- numeric(ndeck)
        for(d in 1:ndeck){
          Ev_curr[d] <- ifelse(d==x[t],
                               Ev_prev[d] + ifelse(X[t]>=0, a_rew, a_pun)*(X[t]-Ev_prev[d]),
                               Ev_prev[d])
        }
        
        # Ef update
        Ef_curr <- numeric(ndeck)
        signX <- ifelse(X[t]>=0, 1, -1)
        for(d in 1:ndeck){
          Ef_cho <- Ef_prev[d] + ifelse(X[t]>=0, a_rew, a_pun)*(signX - Ef_prev[d])
          Ef_not <- Ef_prev[d] + ifelse(X[t]>=0, a_pun, a_rew)*(-(signX/3)-Ef_prev[d])
          Ef_curr[d] <- ifelse(d==x[t], Ef_cho, Ef_not)
        }
        
        # PS update
        PS_curr <- numeric(ndeck)
        for(d in 1:ndeck){
          PS_curr[d] <- ifelse(d==x[t], 1/(1+K), PS_prev[d]/(1+K))
        }
        
        Ev_prev <- Ev_curr
        Ef_prev <- Ef_curr
        PS_prev <- PS_curr
      }
    }
    
    # Store simulated subject data
    sim_data[[s]] <- data.frame(subject=s, trial=1:ntrials,
                                deck=x, reward=X)
  }
  
  return(bind_rows(sim_data))
}

# -------------------------------
# Simulate synthetic subjects
# -------------------------------
nsub_per_group <- 50
n_simulations <- 100 #raise for specificity, lower for testing
simulated_data <- list()

for(g in groups){
  cat("Simulating group:", g, "\n")
  group_sims <- list()
  
  for(i in 1:n_simulations){
    subject_params <- sample_subject_params(param_list[[g]], nsub_per_group)
    sim_data <- simulate_igt(subject_params, ntrials=ntrials, payoffs_matrix=payoffs_matrix)
    sim_data$simulation <- i
    sim_data$group <- g
    group_sims[[i]] <- sim_data
  }
  
  simulated_data[[g]] <- bind_rows(group_sims)
}

# -------------------------------
# Compute behavioral summaries
# -------------------------------
compute_summaries <- function(df){
  df %>%
    group_by(subject, simulation) %>%
    summarise(
      net_score = sum(deck %in% c(3,4), na.rm=TRUE) - sum(deck %in% c(1,2), na.rm=TRUE),
      win_stay = {
        choices <- deck
        reward_vec <- reward
        if(length(choices) > 1){
          stay <- c(NA, choices[-1] == choices[-length(choices)])
          reward_after <- reward_vec[-1]
          mean(stay[-1] & reward_after > 0, na.rm=TRUE)
        } else NA_real_
      },
      lose_shift = {
        choices <- deck
        reward_vec <- reward
        if(length(choices) > 1){
          stay <- c(NA, choices[-1] == choices[-length(choices)])
          reward_after <- reward_vec[-1]
          mean(!stay[-1] & reward_after < 0, na.rm=TRUE)
        } else NA_real_
      },
      .groups='drop'
    )
}

sim_summaries <- lapply(simulated_data, compute_summaries)

# -------------------------------
# Combine summaries for plotting
# -------------------------------
sim_summaries_all <- bind_rows(lapply(names(sim_summaries), function(g){
  df <- sim_summaries[[g]]
  df$group <- g
  df
}))

sim_summaries_long <- sim_summaries_all %>%
  pivot_longer(cols=c(win_stay, lose_shift), names_to="metric", values_to="value")

# -------------------------------
# Plot net score distribution
# -------------------------------
net_score_plot <- ggplot(sim_summaries_all, aes(x=net_score, fill=group)) +
  geom_density(alpha=0.4) +
  # overlay observed net scores
  geom_vline(data=bind_rows(lapply(names(real_data_list), function(g){
    df <- real_data_list[[g]]
    df$group <- g
    df
  })) %>% group_by(group) %>% summarise(net_score=sum(deck %in% c(3,4)) - sum(deck %in% c(1,2))),
  aes(xintercept=net_score, color=group), linetype="dashed", size=1) +
  scale_fill_manual(values=c("ahn_hc"="black","gpt_OLD_2023"="blue","gpt_NEW_2025"="red")) +
  scale_color_manual(values=c("ahn_hc"="black","gpt_OLD_2023"="blue","gpt_NEW_2025"="red")) +
  labs(title="Posterior predictive check: net scores",
       x="Net score (C+D minus A+B)", y="Density") +
  theme_minimal()

ggsave(file.path(ppc_dir, "net_score_ppc.png"), net_score_plot, width=10, height=6, dpi=300)

# -------------------------------
# Plot win-stay / lose-shift
# -------------------------------
ws_ls_plot <- ggplot(sim_summaries_long, aes(x=value, fill=group)) +
  geom_density(alpha=0.4) +
  facet_wrap(~metric, scales="free") +
  scale_fill_manual(values=c("ahn_hc"="black","gpt_OLD_2023"="blue","gpt_NEW_2025"="red")) +
  labs(title="Posterior predictive check: win-stay / lose-shift",
       x="Behavioral measure", y="Density") +
  theme_minimal()

ggsave(file.path(ppc_dir, "ws_ls_ppc.png"), ws_ls_plot, width=10, height=6, dpi=300)


# -------------------------------
# Compute deck choice proportions (real vs simulated)
# -------------------------------

compute_deck_ratios <- function(df){
  df %>%
    group_by(subject, simulation) %>%
    summarise(
      A = mean(deck == 1, na.rm=TRUE),
      B = mean(deck == 2, na.rm=TRUE),
      C = mean(deck == 3, na.rm=TRUE),
      D = mean(deck == 4, na.rm=TRUE),
      .groups='drop'
    ) %>%
    pivot_longer(cols=A:D, names_to="deck", values_to="proportion")
}

# Simulated deck proportions
sim_deck_ratios <- lapply(simulated_data, compute_deck_ratios)
sim_deck_ratios_all <- bind_rows(lapply(names(sim_deck_ratios), function(g){
  df <- sim_deck_ratios[[g]]
  df$group <- g
  df
}))

# Real deck proportions
real_deck_ratios <- lapply(real_data_list, function(df){
  df %>%
    group_by(subject) %>%
    summarise(
      A = mean(deck == 1, na.rm=TRUE),
      B = mean(deck == 2, na.rm=TRUE),
      C = mean(deck == 3, na.rm=TRUE),
      D = mean(deck == 4, na.rm=TRUE),
      .groups='drop'
    ) %>%
    pivot_longer(cols=A:D, names_to="deck", values_to="proportion")
})
real_deck_ratios_all <- bind_rows(lapply(names(real_deck_ratios), function(g){
  df <- real_deck_ratios[[g]]
  df$group <- g
  df
}))

# -------------------------------
# Plot deck choice proportions
# -------------------------------
deck_choice_plot <- ggplot() +
  # Simulated
  geom_violin(data=sim_deck_ratios_all, 
              aes(x=deck, y=proportion, fill=group), alpha=0.4) +
  # Real
  geom_point(data=real_deck_ratios_all, 
             aes(x=deck, y=proportion, color=group), 
             position=position_jitter(width=0.15), size=2) +
  scale_fill_manual(values=c("ahn_hc"="black","gpt_OLD_2023"="blue","gpt_NEW_2025"="red")) +
  scale_color_manual(values=c("ahn_hc"="black","gpt_OLD_2023"="blue","gpt_NEW_2025"="red")) +
  labs(title="Posterior predictive check: deck choice proportions",
       x="Deck", y="Proportion of choices") +
  theme_minimal()

ggsave(file.path(ppc_dir, "deck_choice_ppc.png"), deck_choice_plot, width=10, height=6, dpi=300)

