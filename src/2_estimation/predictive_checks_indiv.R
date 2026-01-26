# Hierarchical ORL model posterior predictive check script (indiv level)
#since indiv level will likely be noisy when dealing with so few participants, do group level instead?

#install.packages("pacman")
pacman::p_load(R2jags, parallel)

# set seed
set.seed(1234)

# defining a function for calculating the maximum of the posterior density
MPD <- function(x) {
  density(x)$x[which(density(x)$y==max(density(x)$y))]
}

# set data grps
groups <- c("ahn_hc", "gpt_EN", "gpt_JP")

### SECTION WHERE YOU CAN CHANGE THINGS ###
fixed_theta <- TRUE # if FALSE then theta is estimated (different model file is used)
model_tag <- ifelse(fixed_theta, "fixedTheta", "freeTheta")

# new root path (name the code folder this)
root_path <- "~/IGT_LLM_DecisionMaking"

# select group
group_name = groups[1] # grp1: human, grp2: gpt_en, grp3: gpt_jp

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

# initialize prediction success array
pred_success <- array(0, c(nsubs))

# set timer
start_time = Sys.time()
print(start_time, "this is when posterior predictive checks started running!")

for (s in 1:nsubs) {
  start_iteration = Sys.time()
  
  # extract data for subject s
  x_s <- x[s, ]
  X_s <- X[s, ]
  ntrials_s <- ntrials[s]
  
  # set up jags data for single subject
  jags_data <- list("x" = x_s, "X" = X_s, "ntrials" = ntrials_s)
  
  if (fixed_theta) {  
    params <- c("a_rew", "a_pun", "K", "omega_f", "omega_p", "deviance", "p") 
    model_file <- file.path(root_path, "models", "ORL_fixed_theta_new_ver.txt")
  } else {
    params <- c("a_rew", "a_pun", "K", "theta", "omega_f", "omega_p", "deviance", "p")
    model_file <- file.path(root_path, "models", "ORL_new_ver.txt")
  }
  
  # run jags model for subject s (should both have inits?)
  temp_samples <- jags.parallel(jags_data, inits=NULL, params,
                                model.file = model_file,
                                n.chains=3, n.iter=3000, n.burnin=1000, n.thin=1, n.cluster=3)
  
  # extract posterior predictions
  p_post <- temp_samples$BUGSoutput$sims.list$p
  
  # generate predictions
  x_predict <- array(0, ntrials_s)
  
  for (j in 1:ntrials_s) {
    p_predict <- c(
      MPD(p_post[,j,1]),
      MPD(p_post[,j,2]),
      MPD(p_post[,j,3]),
      MPD(p_post[,j,4])
    )
    
    x_predict[j] <- which.max(p_predict)
  }
  
  # calculate prediction success
  pred_success[s] <- sum(x_predict == x_s[1:ntrials_s])
  
  end_iteration <- Sys.time()
  run_iteration <- round(end_iteration - start_iteration, 2)
  print(paste0("Subject ", s, " | Iteration time: ", run_iteration, " | Accuracy: ", 
               pred_success[s], "/", ntrials_s))
}

# print total time
end_time = Sys.time()
run_time = round(end_time - start_time, 2)
print(paste0("Total run time: ", run_time))

# make into df with subject IDs
pred_success_df <- data.frame(
  subjID = subIDs,
  pred_success = pred_success,
  ntrials = ntrials,
  accuracy = pred_success / ntrials
)

# save df
savefile <- file.path(root_path, "src", "2_estimation", "results", 
                      paste0("pred_success_", model_tag, "_", group_name, ".csv"))
write.csv(pred_success_df, savefile, row.names = FALSE)

# print summary statistics
cat("\n=== Posterior Predictive Check Summary ===\n")
cat("Group:", group_name, "\n")
cat("Model:", model_tag, "\n")
cat("Mean accuracy:", round(mean(pred_success_df$accuracy), 3), "\n")
cat("SD accuracy:", round(sd(pred_success_df$accuracy), 3), "\n")
cat("Min accuracy:", round(min(pred_success_df$accuracy), 3), "\n")
cat("Max accuracy:", round(max(pred_success_df$accuracy), 3), "\n")