# Setup -------------------------------------------------------------------

install.packages('feather')
library(feather)

X_train <- read_feather('data/X_train.feather')
X_test <- read_feather('data/X_test.feather')
X_pred <- read_feather('data/X_pred.feather')

library(h2o)
n_seed = 888

# Launch H2O --------------------------------------------------------------

h2o.init()
h2o.removeAll()

# Analyze cleaned data ----------------------------------------------------

target = 'SalePrice'

X <-rbind(X_train, X_test)

features = setdiff(colnames(X), target)

X.h2o = as.h2o(X)

X.h2o.split = h2o.splitFrame(X.h2o, ratios=0.8, seed=n_seed)
X.h2o.train = X.h2o.split[[1]]
X.h2o.test = X.h2o.split[[2]]

mod.auto = h2o.automl(x = features,
                      y = target,
                      training_frame = X.h2o.train,
                      nfolds = 5,               # Cross-Validation
                      max_runtime_secs = 30,   # Max time
                      max_models = 100,         # Max no. of models
                      stopping_metric = "RMSLE", # Metric to optimize
                      project_name = "automl_reg",
                      exclude_algos = NULL,     # If you want to exclude any algo 
                      seed = n_seed)

mod.auto@leaderboard
mod.auto@leader

h2o.performance(mod.auto@leader, newdata = X.h2o.test)

X.h2o.pred = as.h2o(X_pred)

# THIS IS WHERE IT FAILS
pred = h2o.predict(mod.auto, X.h2o.pred)

res <- cbind(X_pred$Id, as.data.frame(exp(pred)))
colnames(res) <- c('Id', 'SalePrice')

library(utils)
write.csv(res, 'submission.csv', row.names = F)

h2o.shutdown()