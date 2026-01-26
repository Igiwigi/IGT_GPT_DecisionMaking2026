#given that it's a poor fit for our LLM data, leaving free theta unfinished (it doesnt quite work as it should, though it did... for a moment?)
#we care about group level estimates and subject variability!!!!

# Hierarchical ORL model estimation script
# using new version of the code and models! (redundant to have 2 codes but most convenient rn)


#install.packages("pacman")
pacman::p_load(R2jags, parallel)

# set seed
set.seed(1234)

#set data grps
groups <- c("ahn_hc", "gpt_OLD_2023", "gpt_NEW_2025")

### SECTION WHERE YOU CAN CHANGE THINGS ###
fixed_theta <- TRUE # if FALSE then theta is estimated (included in params and a different model file is used)
model_tag <- ifelse(fixed_theta, "fixedTheta", "freeTheta")

#new root path (name the code folder this)
root_path <- "~/IGT_LLM_DecisionMaking"

# select group and read in data 
group_name = groups[1] # grp1: human, grp2: gpt_old, grp3: gpt_new

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

#set some inits so the chains dont mix poorly (hopefully)
inits <- function() {
  list(
    mu_a_rew = rnorm(1, -1.5, 0.3),
    mu_a_pun = rnorm(1, -2.5, 0.3),
    mu_K = rnorm(1, 0, 0.3),
    mu_omega_f = rnorm(1, 0, 0.5),
    mu_omega_p = rnorm(1, 0, 0.5),
    
    sigma_a_rew = runif(1, 0.3, 0.8),
    sigma_a_pun = runif(1, 0.3, 0.8),
    sigma_K = runif(1, 0.3, 0.8),
    sigma_omega_f = runif(1, 0.5, 1.5),
    sigma_omega_p = runif(1, 0.5, 1.5),
    
    z_a_rew = rnorm(nsubs, 0, 0.5),
    z_a_pun = rnorm(nsubs, 0, 0.5),
    z_K = rnorm(nsubs, 0, 0.5),
    z_of = rnorm(nsubs, 0, 0.5),
    z_op = rnorm(nsubs, 0, 0.5)
  )
}

# set timer
start_time = Sys.time()
print(start_time, "this is when this started running!") #too lazy to fix the error regarding the timezone

# run jags model
print("Intializing JAGS ...")
print(paste("Running model:", model_tag, "| Group:", group_name))

#doing high iter. to make up for having only 30 data points
#samples <- jags.parallel(jags_data, inits=inits, params,
#                         model.file = model_file,
#                         n.chains=3, n.iter=10000, n.burnin=1500, n.thin=2, n.cluster=4, jags.seed = 1234)

#what was here originally
samples <- jags.parallel(jags_data, inits=NULL, params,
                         model.file = model_file,
                        n.chains=3, n.iter=3000, n.burnin=1000, n.thin=1, n.cluster=4, jags.seed= 1234)

#doesnt print with bad deviance (if doesnt converge)
print(samples$BUGSoutput$summary)
print(samples$BUGSoutput$DIC)

# save summary to txt file depending on group
summary_file <- file.path(root_path, "src", "2_estimation", "results", "new_model",
                          paste0("summary_grouplevel_", model_tag, "_", group_name, ".txt"))
write.table(samples$BUGSoutput$summary, summary_file)

# save DIC to separate file
dic_file <- file.path(root_path, "src", "2_estimation", "results", "new_model",
                      paste0("DIC_grouplevel_", model_tag, "_", group_name, ".txt"))
write.table(samples$BUGSoutput$DIC, dic_file)

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
plotfile <- file.path(root_path, "src", "2_estimation", "plots", "traceplots", "new_model",
                      paste0("trace_", model_tag, "_", group_name, ".png"))

# Calculate appropriate grid dimensions based on number of parameters (since it varies)
n_params <- length(params) - 1 #exclude deviance
n_cols <- 2
n_rows <- ceiling(n_params / n_cols)

png(plotfile, width = 2500, height = 800 * n_rows, res = 400)
traceplot(samples$BUGSoutput, ask=FALSE, mfrow = c(n_rows, n_cols), varname=params[params != "deviance"])
dev.off()

#plot deviance separately
deviance_plotfile <- file.path(root_path, "src", "2_estimation", "plots", "traceplots", "new_model",
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

# save df
savefile <- file.path(root_path, "src", "2_estimation", "results", "new_model",
                      paste0("grouplevel_", model_tag, "_", group_name, ".csv"))

write.csv(df, savefile, row.names = FALSE)

# print time run
end_time = Sys.time()
print(end_time - start_time)