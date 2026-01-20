#still issues with theta!!

#install.packages("pacman")
pacman::p_load(R2jags, parallel)

# set seed
set.seed(2502)

# overview of all groups that can be run 
#groups <- c("gpt", "ahn_hc", "ahn_hc_1", "ahn_hc_2", "ahn_hc_3", "ahn_hc_4")

#dont need to use the extra sample groups (yet)
groups <- c("ahn_hc", "gpt_EN", "gpt_JP")

### SECTION WHERE YOU CAN CHANGE THINGS ###
fixed_theta <- FALSE # if FALSE then theta is estimated (included in params and a different model file is used)
model_tag <- ifelse(fixed_theta, "fixedTheta", "freeTheta") #to know which

# change the root path according to your needs (i.e., where the main folder is located)
#root_path <- "~/Desktop/dm-code" # personal comp
#root_path <- "dm-code" # UCloud

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
  params<-c("mu_a_rew","mu_a_pun","mu_K","mu_omega_f","mu_omega_p","deviance") 
  model_file <- file.path(root_path, "models", "hier_ORL_fixed_theta.txt")
} else {
  params<-c("mu_a_rew","mu_a_pun","mu_K","mu_theta","mu_omega_f","mu_omega_p","deviance") 
  model_file <- file.path(root_path, "models", "hier_ORL_new_uninformed_priors.txt") #testing a new hier ORL! (less overflow, still uninformed priors)
}

# set timer
start_time = Sys.time()
# run jags model
print("Intializing JAGS ...")
print(paste("Running model:", model_tag, "| Group:", group_name))

#can maybe run the model for longer iters (though likely structural issues)
#test n.iter 10000 later with n.burnin 3000
samples <- jags.parallel(jags_data, inits=NULL, params,
                         model.file = model_file,
                         n.chains=3, n.iter=3000, n.burnin=1000, n.thin=1, n.cluster=4)

end_time <- Sys.time()
print(paste("Runtime:", end_time - start_time, "took this long :)"))

#check if NA's
print(any(is.na(samples$BUGSoutput$sims.list$mu_theta)))
print(any(is.infinite(samples$BUGSoutput$sims.list$mu_theta)))

#doesnt print with bad deviance
if(fixed_theta) {
  print(samples$BUGSoutput)
} else {
  print(samples$BUGSoutput$summary)
}

# save summary to txt file depending on group
summary_file <- file.path(root_path, "src", "2_estimation", "results",
                          paste0("summary_", model_tag, "_", group_name, ".txt"))

write.table(samples$BUGSoutput$summary, summary_file)

# extact mu parameters
Y <- samples$BUGSoutput$sims.list
mu_a_rew <- Y$mu_a_rew
mu_a_pun <- Y$mu_a_pun
mu_K <- Y$mu_K
if (!fixed_theta) {
  mu_theta <- Y$mu_theta
}
mu_omega_f <- Y$mu_omega_f
mu_omega_p <- Y$mu_omega_p

# plot traceplots
plotfile <- file.path(root_path, "src", "2_estimation", "plots", "traceplots",
                      paste0("trace_", model_tag, "_", group_name, ".png"))
png(plotfile, width = 2500, height = 2500, res = 400)
traceplot(samples$BUGSoutput, ask=FALSE, mfrow = c(3, 2), varname=params)
dev.off()

# save to df 
if (fixed_theta) {
  df <- data.frame(mu_a_rew, mu_a_pun, mu_K, mu_omega_f, mu_omega_p)
} else {
 df <- data.frame(mu_a_rew, mu_a_pun, mu_K, mu_theta, mu_omega_f, mu_omega_p)}


# save df (if it is an extra sample e.g., ahn_hc_3, then save to the "extra_samples" folder within results)
savefile <- file.path(root_path, "src", "2_estimation", "results",
                      paste0("param_", model_tag, "_", group_name, ".csv"))

write.csv(df, savefile)

# print time run
end_time = Sys.time()
print(end_time - start_time)