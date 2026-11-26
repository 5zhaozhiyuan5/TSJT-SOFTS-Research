# 从仓库根目录运行: Rscript data/zero_inflated/scripts/export_saeczi.R
base <- file.path("data", "zero_inflated", "saeczi")
raw_dir <- file.path(base, "raw")
out_dir <- file.path(base, "exported")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
for (name in c("samp", "pop")) {
  file <- file.path(raw_dir, paste0(name, ".rda"))
  if (!file.exists(file)) stop("Missing data file: ", file)
  e <- new.env(parent = baseenv())
  objects <- load(file, envir = e)
  if (!(name %in% objects)) stop("Expected R object '", name, "' not found in ", file)
  dat <- get(name, envir = e)
  csv <- file.path(out_dir, paste0(name, ".csv"))
  write.csv(dat, csv, row.names = FALSE, fileEncoding = "UTF-8")
  cat(name, ": ", nrow(dat), " rows × ", ncol(dat), " cols; ", csv, "\n", sep = "")
  if (name == "samp" && "DRYBIO_AG_TPA_live_ADJ" %in% names(dat)) {
    y <- dat$DRYBIO_AG_TPA_live_ADJ
    good <- is.finite(y)
    if (any(good)) cat("  zero_ratio = ", round(mean(y[good] == 0), 5), "\n", sep = "")
  }
}
