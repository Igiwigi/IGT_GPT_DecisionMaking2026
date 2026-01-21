#still issues with theta (when unfixed), likely doesnt work as it should, and likely will not be fixed as it will not be used!! 
#given that it's a poor fit for our LLM data anyway, leaving free theta unfinished (it doesnt quite work as it should, though it did... for a moment.)
#likely would need better inits for a theta model (so init values cant ruin it) & bad rhat anyway

#we care about group level estimates and subject variability!!!!

#install.packages("pacman")
pacman::p_load(R2jags, parallel)

# set seed
set.seed(2502)

#set data grps
groups <- c("ahn_hc", "gpt_EN", "gpt_JP")

### SECTION WHERE YOU CAN CHANGE THINGS ###
fixed_theta <- TRUE # if FALSE then theta is estimated (included in params and a different model file is used)
model_tag <- ifelse(fixed_theta, "fixedTheta", "freeTheta")

#new root path (name the code folder this)
root_path <- "~/IGT_LLM_DecisionMaking"

# select group and read in data 
group_name = groups[2] # grp1: human, grp2: gpt_en, grp3: gpt_jp

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
x <- array(0,c(nsubs,ntrials_max))
X <- array(0,c(nsubs,ntrials_max))
ntrials <- array(0,c(nsubs))

# turn data from long format into arrays with (nsubs x ntrials_max) dimensions

print(paste0("Working on file: ", basename(file)))
for (s in 1:nsubs) {
  
  #record n trials for subject s
  ntrials[s] <- length(x_raw[data$subjID==subIDs[s]])
  
  # extract x and X for subject s
  X_sub <- X_raw[data$subjID==subIDs[s]] 
  x_sub <- x_raw[data$subjID==subIDs[s]] 
  
  # assign arrays
  x[s,] <- x_sub
  X[s,] <- X_sub
}

# scale the payoffs
X <- X/100

# set up jags and run jags model
jags_data <- list("x","X","ntrials","nsubs") 
if (fixed_theta) {
  params <- c("mu_a_rew", "mu_a_pun", "mu_K", "mu_omega_f", "mu_omega_p",
              "sigma_a_rew", "sigma_a_pun", "sigma_K", "sigma_omega_f", "sigma_omega_p",
              "deviance") #sigma (subj lvl variability), mu (grp lvl variability) (just incase we want to report on both)
  model_file <- file.path(root_path, "models", "hier_ORL_fixed_theta_new_ver.txt")
  #testing hopefully a less error-prone version (some problems with vars and the way theyre bounded & initialized, less centered now)
} else {
  params <- c("mu_a_rew", "mu_a_pun", "mu_logK", "mu_logtheta", "mu_omega_f", "mu_omega_p",
              "sigma_a_rew", "sigma_a_pun", "sigma_K", "sigma_theta", "sigma_omega_f", "sigma_omega_p",
              "deviance")
  model_file <- file.path(root_path, "models", "hier_ORL_new_ver.txt") 
  #runs, but poor convergence on variables and terrible rhat (also, ungodly slow) 
  #will not be using in the end
}

# set timer
start_time = Sys.time()

# run jags model
print("Intializing JAGS ...")
print(paste("Running model:", model_tag, "| Group:", group_name))

#doing high iter to make up for having little data (16 participants)
samples <- jags.parallel(jags_data, inits=NULL, params,
                         model.file = model_file,
                         n.chains=3, n.iter=20000, n.burnin=8000, n.thin=5, n.cluster=4)

#samples <- jags.parallel(jags_data, inits=NULL, params,
#                         model.file = model_file,
#                         n.chains=3, n.iter=3000, n.burnin=1000, n.thin=1, n.cluster=4)

end_time <- Sys.time()
print(paste("Runtime:", end_time - start_time, "took this long :)"))

#doesnt print with bad deviance
print(samples$BUGSoutput$summary)

# save summary to txt file depending on group
summary_file <- file.path(root_path, "src", "2_estimation", "results",
                          paste0("summary_", model_tag, "_", group_name, ".txt"))

write.table(samples$BUGSoutput$summary, summary_file)

# extract group-level parameters (means and SDs)
Y <- samples$BUGSoutput$sims.list

# Group-level means
mu_a_rew <- Y$mu_a_rew
mu_a_pun <- Y$mu_a_pun
mu_omega_f <- Y$mu_omega_f
mu_omega_p <- Y$mu_omega_p

# Between-subject variability (SDs)
sigma_a_rew <- Y$sigma_a_rew
sigma_a_pun <- Y$sigma_a_pun
sigma_K <- Y$sigma_K
sigma_omega_f <- Y$sigma_omega_f
sigma_omega_p <- Y$sigma_omega_p

if (fixed_theta) {
  # For fixed theta model, mu_K is directly available
  mu_K <- Y$mu_K
} else {
  # For free theta model, extract log-scale parameters and transform to natural scale
  #might not work as it should
  mu_logK <- Y$mu_logK
  mu_K <- exp(mu_logK)
  
  mu_logtheta <- Y$mu_logtheta
  mu_theta <- exp(mu_logtheta)
  
  sigma_theta <- Y$sigma_theta
}

# plot traceplots
plotfile <- file.path(root_path, "src", "2_estimation", "plots", "traceplots",
                      paste0("trace_", model_tag, "_", group_name, ".png"))

# Calculate appropriate grid dimensions based on number of parameters (since it varies)
n_params <- length(params)
n_cols <- 2
n_rows <- ceiling(n_params / n_cols)

png(plotfile, width = 2500, height = 800 * n_rows, res = 400)
traceplot(samples$BUGSoutput, ask=FALSE, mfrow = c(n_rows, n_cols), varname=params[params != "deviance"])
dev.off()

#plot deviance separately
deviance_plotfile <- file.path(root_path, "src", "2_estimation", "plots", "traceplots",
                               paste0("trace_deviance_", model_tag, "_", group_name, ".png"))
png(deviance_plotfile, width = 2500, height = 1200, res = 400)
traceplot(samples$BUGSoutput, ask=FALSE, mfrow = c(1, 1), varname="deviance")
dev.off()

# save to df with both means and SDs
if (fixed_theta) {
  df <- data.frame(mu_a_rew, mu_a_pun, mu_K, mu_omega_f, mu_omega_p,
                   sigma_a_rew, sigma_a_pun, sigma_K, sigma_omega_f, sigma_omega_p)
} else {
  df <- data.frame(mu_a_rew, mu_a_pun, mu_K, mu_theta, mu_omega_f, mu_omega_p,
                   sigma_a_rew, sigma_a_pun, sigma_K, sigma_theta, sigma_omega_f, sigma_omega_p)
}

# save df (if it is an extra sample e.g., ahn_hc_3, then save to the "extra_samples" folder within results; not implemented in this version rn)
savefile <- file.path(root_path, "src", "2_estimation", "results",
                      paste0("grouplevel_", model_tag, "_", group_name, ".csv"))

write.csv(df, savefile, row.names = FALSE)

# print time run
end_time = Sys.time()
print(end_time - start_time)