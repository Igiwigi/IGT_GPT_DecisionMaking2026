#testing simplified models
#why does this not provide a proper traceback???

#install.packages("pacman")
pacman::p_load(R2jags, parallel)

# set seed
set.seed(2502)

# Groups that can be run
groups <- c("ahn_hc", "gpt_EN", "gpt_JP")

### SECTION WHERE YOU CAN CHANGE THINGS ###
fixed_theta <- FALSE # if FALSE then theta is estimated
model_tag <- ifelse(fixed_theta, "fixedTheta_simplified", "freeTheta_simplified")

# Root path to your project
root_path <- "~/IGT_LLM_DecisionMaking"

# Select group and read in data 
group_name = groups[1] # group1 human, group2 gpt_en, group3 gpt_jp
file <- file.path(root_path, "data", "final_data", paste0("clean_", group_name, ".csv"))
data <- read.csv(file)

# Extract relevant variables
subIDs <- unique(data$subjID)
nsubs <- length(subIDs)
x_raw <- data$x
X_raw <- data$X

# Initialize empty arrays
ntrials_max <- 100
x <- array(0,c(nsubs,ntrials_max))
X <- array(0,c(nsubs,ntrials_max))
ntrials <- array(0,c(nsubs))

# Convert long-format to arrays
print(paste0("Working on file: ", basename(file)))
for (s in 1:nsubs) {
  ntrials[s] <- length(x_raw[data$subjID==subIDs[s]])
  X_sub <- X_raw[data$subjID==subIDs[s]] 
  x_sub <- x_raw[data$subjID==subIDs[s]] 
  x[s,] <- x_sub
  X[s,] <- X_sub
}

# Scale the payoffs
X <- X / 100

# Set up JAGS data
jags_data <- list("x","X","ntrials","nsubs") 

# Set model file and parameters based on fixed/free theta
if (fixed_theta) {
  params <- c("mu_alpha","mu_omega_f","mu_omega_p","deviance")
  model_file <- file.path(root_path, "models", "simplified_hier_ORL_fixed_theta.txt")
} else {
  params <- c("mu_alpha","mu_theta","mu_omega_f","mu_omega_p","deviance")
  model_file <- file.path(root_path, "models", "simplified_hier_ORL_for_near-ideal_players.txt")
}

# Run JAGS
start_time <- Sys.time()
print("Initializing JAGS ...")
print(paste("Running model:", model_tag, "| Group:", group_name))

samples <- jags.parallel(jags_data, inits=NULL, params,
                         model.file = model_file,
                         n.chains=3, n.iter=3000, n.burnin=1000, n.thin=1, n.cluster=4)

end_time <- Sys.time()
print(paste("Runtime:", end_time - start_time))

# Print summary
print(samples$BUGSoutput$summary)

# Save summary to txt
summary_file <- file.path(root_path, "src", "2_estimation", "results",
                          paste0("summary_", model_tag, "_", group_name, ".txt"))
write.table(samples$BUGSoutput$summary, summary_file)

# Extract mu parameters
Y <- samples$BUGSoutput$sims.list
mu_alpha <- Y$mu_alpha
mu_omega_f <- Y$mu_omega_f
mu_omega_p <- Y$mu_omega_p
if (!fixed_theta) {
  mu_theta <- Y$mu_theta
}

# Plot traceplots
plotfile <- file.path(root_path, "src", "2_estimation", "plots", "traceplots",
                      paste0("trace_", model_tag, "_", group_name, ".png"))
png(plotfile, width = 2500, height = 2500, res = 400)
if (fixed_theta) {
  traceplot(samples$BUGSoutput, ask=FALSE, mfrow = c(2,2), varname=params)
} else {
  traceplot(samples$BUGSoutput, ask=FALSE, mfrow = c(2,2), varname=params)
}
dev.off()

# Save to df
if (fixed_theta) {
  df <- data.frame(mu_alpha, mu_omega_f, mu_omega_p)
} else {
  df <- data.frame(mu_alpha, mu_theta, mu_omega_f, mu_omega_p)
}

# Save CSV
savefile <- file.path(root_path, "src", "2_estimation", "results",
                      paste0("param_", model_tag, "_", group_name, ".csv"))
write.csv(df, savefile)

# Print runtime
print(end_time - start_time)
