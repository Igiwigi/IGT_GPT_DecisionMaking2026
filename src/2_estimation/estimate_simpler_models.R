# Simple Models Estimation Script (WSLS & Softmax)
# Adapted for Win-Stay-Lose-Switch and Softmax models

#install.packages("pacman")
pacman::p_load(R2jags, parallel)

# set seed
set.seed(2502)

# set data groups
groups <- c("ahn_hc", "gpt_EN", "gpt_JP")

### SECTION WHERE YOU CAN CHANGE THINGS ###
WSLS <- FALSE # if TRUE runs WSLS, if FALSE runs Softmax
model_tag <- ifelse(WSLS, "WSLS", "Softmax")

# new root path (name the code folder this)
root_path <- "~/IGT_LLM_DecisionMaking"

# select group and read in data 
group_name <- groups[3] # grp1: human, grp2: gpt_en, grp3: gpt_jp

### CODE ###
# create and load file based on group selection
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

# turn data from long format into arrays with (nsubs x ntrials_max) dimensions
print(paste0("Working on file: ", basename(file)))
for (s in 1:nsubs) {
  
  # record n trials for subject s
  ntrials[s] <- length(x_raw[data$subjID == subIDs[s]])
  
  # extract x and X for subject s
  X_sub <- X_raw[data$subjID == subIDs[s]] 
  x_sub <- x_raw[data$subjID == subIDs[s]] 
  
  # assign arrays
  x[s, ] <- x_sub
  X[s, ] <- X_sub
}

# scale the payoffs
X <- X / 100

# set up jags and run jags model
jags_data <- list("x", "X", "ntrials", "nsubs") 

# Set parameters and model file based on model type
if (WSLS) {
  params <- c("mu_prob_stay_win", "mu_prob_stay_lose", "mu_epsilon", "deviance") 
  model_file <- file.path(root_path, "models", "simpler_models", "WSLS_hier.txt")
} else {
  params <- c("mu_alpha", "mu_theta", "deviance")
  model_file <- file.path(root_path, "models", "simpler_models", "Softmax_hier.txt")
}

# set timer
start_time <- Sys.time()

# run jags model
print("Initializing JAGS ...")
print(paste("Running model:", model_tag, "| Group:", group_name))

samples <- jags.parallel(jags_data, inits = NULL, params,
                         model.file = model_file,
                         n.chains = 3, n.iter = 3000, n.burnin = 1000, 
                         n.thin = 1, n.cluster = 4)

end_time <- Sys.time()
print(paste("Runtime:", end_time - start_time))

# print summary
print(samples$BUGSoutput$summary)

# save summary to txt file
summary_file <- file.path(root_path, "src", "2_estimation", "results",
                          paste0("summary_", model_tag, "_", group_name, ".txt"))

write.table(samples$BUGSoutput$summary, summary_file)

# extract mu parameters and create df based on model type
Y <- samples$BUGSoutput$sims.list

if (WSLS) {
  mu_prob_stay_win <- Y$mu_prob_stay_win
  mu_prob_stay_lose <- Y$mu_prob_stay_lose
  mu_epsilon <- Y$mu_epsilon
  df <- data.frame(mu_prob_stay_win, mu_prob_stay_lose, mu_epsilon)
} else {
  mu_alpha <- Y$mu_alpha
  mu_theta <- Y$mu_theta
  df <- data.frame(mu_alpha, mu_theta)
}

# plot traceplots
plotfile <- file.path(root_path, "src", "2_estimation", "plots", "traceplots",
                      paste0("trace_", model_tag, "_", group_name, ".png"))
png(plotfile, width = 2500, height = 2500, res = 400)
traceplot(samples$BUGSoutput, ask = FALSE, mfrow = c(2, 2), varname = params)
dev.off()

# save df
savefile <- file.path(root_path, "src", "2_estimation", "results", "simpler_models",
                      paste0("param_", model_tag, "_", group_name, ".csv"))

write.csv(df, savefile)

# print final runtime
print(paste("Total runtime:", end_time - start_time))