# Simple Models Estimation Script (WSLS & Softmax)
# Adapted for Win-Stay-Lose-Switch and Softmax models (WSLS is used in the paper)

pacman::p_load(R2jags, parallel)

# set seed
set.seed(1234)

# set data groups
groups <- c("ahn_hc", "gpt_OLD_2023", "gpt_NEW_2025")

### SECTION WHERE YOU CAN CHANGE THINGS ###
WSLS <- TRUE # if TRUE runs WSLS, if FALSE runs Softmax (SOFTMAX is not used, just tested)
model_tag <- ifelse(WSLS, "WSLS", "Softmax")

# new root path (name the code folder this)
root_path <- "~/IGT_LLM_DecisionMaking"
root_path <- "C:/Users/Igiba/Documents/decision_making_course/IGT_GPT_DecisionMaking2026" #on laptop, too lazy to fix

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
x <- array(0, c(nsubs, ntrials_max))
X <- array(0, c(nsubs, ntrials_max))
ntrials <- array(0, c(nsubs))

# turn data from long format into arrays
print(paste0("Working on file: ", basename(file)))
for (s in 1:nsubs) {
  ntrials[s] <- length(x_raw[data$subjID == subIDs[s]])
  X[s, ] <- X_raw[data$subjID == subIDs[s]] 
  x[s, ] <- x_raw[data$subjID == subIDs[s]] 
}

# scale the payoffs
X <- X / 100

# set up jags data
jags_data <- list("x", "X", "ntrials", "nsubs") 

# parameters and model file
if (WSLS) {
  params <- c(
    "mu_prob_stay_win", "mu_prob_switch_lose",
    "lambda_prob_stay_win", "lambda_prob_switch_lose",
    "prob_stay_win", "prob_switch_lose",
    "x_pred", "deviance" # -> NO EPSILON, WHY WAS IT EVER THERE TO BEGIN WITH?
  )
  model_file <- file.path(root_path, "models", "simpler_models", "2nd_try", "WSLS_hier_no_epsilon.txt")
} else {
  params <- c(
    "mu_alpha", "mu_theta", 
    "lambda_alpha", "lambda_theta",
    "alpha", "theta",
    "x_pred", "deviance"
  )
  model_file <- file.path(root_path, "models", "simpler_models", "Softmax_hier.txt")
}

# timer
start_time <- Sys.time()

# run jags
print("Initializing JAGS ...")
print(paste("Running model:", model_tag, "| Group:", group_name))

samples <- jags.parallel(
  data = jags_data,
  inits = NULL,
  parameters.to.save = params,
  model.file = model_file,
  n.chains = 3,
  n.iter = 10000, #i upped all this to have slightly better convergence overall (will also be reflected in all plots after this)
  n.burnin = 2000,
  n.thin = 1,
  n.cluster = 4,
  jags.seed= 1234
)

end_time <- Sys.time()
print(paste("Runtime:", end_time - start_time))

#make the summary cleaner (dont spam with param[i])
summary_full <- samples$BUGSoutput$summary
keep_rows <- !grepl("\\[", rownames(summary_full))
summary_clean <- summary_full[keep_rows, ]

# print clean summary
print(summary_clean)

# save clean summary
summary_file <- file.path(
  root_path, "src", "2_estimation", "results", "simpler_models",
  paste0("summary_", model_tag, "_", group_name, ".txt")
)
write.table(summary_clean, summary_file)

# extract posterior samples
Y <- samples$BUGSoutput$sims.list

if (WSLS) {
  df <- data.frame(
    mu_prob_stay_win = Y$mu_prob_stay_win,
    mu_prob_switch_lose = Y$mu_prob_switch_lose,
    lambda_prob_stay_win = Y$lambda_prob_stay_win,
    lambda_prob_switch_lose = Y$lambda_prob_switch_lose,
  )
} else {
  df <- data.frame(
    mu_alpha = Y$mu_alpha,
    mu_theta = Y$mu_theta,
    lambda_alpha = Y$lambda_alpha,
    lambda_theta = Y$lambda_theta
  )
}

# traceplots (group-level only)
plotfile <- file.path(
  root_path, "src", "2_estimation", "convergence_plots", "simpler_models",
  paste0("trace_", model_tag, "_", group_name, ".png")
)

if (WSLS) {
  trace_params <- c(
    "mu_prob_stay_win",
    "mu_prob_switch_lose",
    "deviance"
  )
} else {
  trace_params <- c("mu_alpha", "mu_theta")
}

png(plotfile, width = 2500, height = 2500, res = 400)
traceplot(
  samples$BUGSoutput,
  ask = FALSE,
  mfrow = c(2, 2),
  varname = trace_params
)
dev.off()

# save parameter df
savefile <- file.path(
  root_path, "src", "2_estimation", "results", "simpler_models",
  paste0("param_estimated_", model_tag, "_", group_name, ".csv")
)
write.csv(df, savefile, row.names = FALSE)

# save DIC
dic_file <- file.path(
  root_path, "src", "2_estimation", "results", "simpler_models",
  paste0("DIC_grouplevel_", model_tag, "_", group_name, ".txt")
)
write.table(samples$BUGSoutput$DIC, dic_file)

# save full fit
fitfile <- file.path(
  root_path, "src", "2_estimation", "results", "simpler_models",
  paste0("jags_fit_", model_tag, "_", group_name, ".RData")
)
save(samples, file = fitfile)

print(paste("JAGS fit saved to:", fitfile))
print(paste("Total runtime:", end_time - start_time))
