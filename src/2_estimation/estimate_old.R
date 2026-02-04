#given that it's a poor fit for our LLM data, leaving free theta unfinished (it doesnt quite work as it should, though it did... for a moment?)
#we care about group level estimates and subject variability!!!!

# Hierarchical ORL model estimation script 
# using old/original version of the code and models!
# (testing this to see if I can reject it later)

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
group_name = groups[3] # grp1: human (30), grp2: gpt_old (30), grp3: gpt_new (32)

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
  params<-c("mu_a_rew","mu_a_pun","mu_K","mu_omega_f","mu_omega_p", "deviance") 
  model_file <- file.path(root_path, "models", "hier_ORL_fixed_theta.txt")
} else {
  params<-c("mu_a_rew","mu_a_pun","mu_K","mu_theta","mu_omega_f","mu_omega_p", "deviance") 
  model_file <- file.path(root_path, "models", "hier_ORL.txt")
}

# set timer
start_time = Sys.time()

# run jags model
print("Intializing JAGS ...")
samples <- jags.parallel(jags_data, inits=NULL, params,
                         model.file = model_file,
                         n.chains=3, n.iter=3000, n.burnin=1000, n.thin=1, n.cluster=4, jags.seed = 1234)

print(samples$BUGSoutput$summary)
print(samples$BUGSoutput$DIC)

# save summary to txt file depending on group (clumsily saving all as fixedTheta rn since none are non-fixed)
ifelse(grepl("^ahn_hc_", group_name),
       summary_file <- file.path(root_path, "src", "2_estimation", "results", "extra_samples", paste0("summary_param_estimated_fixedTheta_", group_name, ".txt")),
       summary_file <- file.path(root_path, "src", "2_estimation", "results", "old_model", paste0("summary_param_estimated_fixedTheta_", group_name, ".txt")))

write.table(samples$BUGSoutput$summary, summary_file)

# save DIC to separate file
dic_file <- file.path(root_path, "src", "2_estimation", "results", "old_model",
                      paste0("DIC_grouplevel_", model_tag, "_", group_name, ".txt"))
write.table(samples$BUGSoutput$DIC, dic_file)

#plot deviance separately
deviance_plotfile <- file.path(root_path, "src", "2_estimation", "plots", "old_Model",
                               paste0("trace_deviance_", model_tag, "_", group_name, ".png"))
png(deviance_plotfile, width = 2500, height = 1200, res = 400)
traceplot(samples$BUGSoutput, ask=FALSE, mfrow = c(1, 1), varname="deviance")
dev.off()


# extact mu parameters
Y <- samples$BUGSoutput$sims.list
mu_a_rew <- Y$mu_a_rew
mu_a_pun <- Y$mu_a_pun
mu_K <- Y$mu_K
mu_theta <- Y$mu_theta
mu_omega_f <- Y$mu_omega_f
mu_omega_p <- Y$mu_omega_p

# plot traceplots
plotfile <- file.path(root_path, "src", "2_estimation", "convergence_plots", "old_model", paste0("trace_", group_name, ".png"))
png(plotfile, width = 2500, height = 2500, res = 400)
traceplot(samples$BUGSoutput, ask=FALSE, mfrow = c(3, 2), varname=params)
dev.off()

# save to df 
if (fixed_theta) {
  df <- data.frame(mu_a_rew, mu_a_pun, mu_K, mu_omega_f, mu_omega_p)
} else {
  df <- data.frame(mu_a_rew, mu_a_pun, mu_K, mu_theta, mu_omega_f, mu_omega_p)
}

# save df (if it is an extra sample e.g., ahn_hc_3, then save to the "extra_samples" folder within results)
ifelse(grepl("^ahn_hc_", group_name),
       savefile <- file.path(root_path, "src", "2_estimation", "results", "extra_samples", paste0("param_estimated_fixedTheta_", group_name, ".csv")),
       savefile <- file.path(root_path,"src", "2_estimation", "results", "old_model", paste0("param_estimated_fixedTheta_", group_name, ".csv")))

write.csv(df, savefile)

# print time run
end_time = Sys.time()
print(end_time - start_time)