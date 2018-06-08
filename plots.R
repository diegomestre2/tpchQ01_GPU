library(dplyr)
library(ggplot2)
library(ggthemes)
library(ggrepel)
library(stringr)
library(grid)
library(reshape2)


theme <- theme_few(base_size = 24) + 
theme(axis.title.y=element_text(vjust=0.9), 
  axis.title.x=element_text(vjust=-0.1),
  axis.ticks.x=element_blank(),
  text=element_text(family="serif"),
  legend.position="none")

system('mkdir -p plots')

# SF plots
sfs <- c(1, 10, 100)
for(sf in sfs) {
    data <- read.table(paste0("results/sf/results-sf", sf, ".csv"), header=F,  sep=",", stringsAsFactors=F, na.strings="-1")
    names(data) <- c("time")
    data$sf <- sf
    if (sf == sfs[1]) {
        total_data <- data
    } else {
        total_data <- rbind(total_data, data)
    }
}

total_data %>% group_by(sf) %>% summarise(time=median(time)) %>% as.data.frame() -> total_data

pdf("plots/sf.pdf", width=8, height=6)
ggplot(total_data, aes(x = sf, y = time)) + geom_line() + theme + xlab("Scale Factor (#)") + ylab("Wall clock time (s)") + geom_point() + scale_x_log10(breaks=sfs) + scale_y_log10()
dev.off()

# Streams plots
streams <- c(1, 2, 4, 8, 16, 32, 64, 128, 256)
tuples <- c(1024, 2*1024, 4*1024, 8*1024, 16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 1024*1024)

for(stream in streams) {
    for(tpls in tuples) {
        data <- read.table(paste0("results/streams/results-streams", stream, "-tuples", tpls, ".csv"), header=F,  sep=",", stringsAsFactors=F, na.strings="-1")
        names(data) <- c("time")
        data$tuples <- tpls
        data$stream <- stream
        if (stream == streams[1] && tpls == tuples[[1]]) {
            total_data <- data
        } else {
            total_data <- rbind(total_data, data)
        }
    }
}
total_data %>% group_by(tuples, stream) %>% summarise(time=median(time)) %>% as.data.frame() -> total_data

pdf("plots/streams.pdf", width=7, height=7)
ggplot(data = total_data, aes(x = stream, y = tuples)) +
  geom_tile(aes(fill = time)) + xlab("Total Streams (#)") + ylab("Tuples Per Streams (#)") +
   scale_x_continuous(trans='log2', breaks=streams) + scale_y_continuous(trans='log2', breaks=tuples)
dev.off()

# threads, values
values_per_thread = c(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
threads_per_block = c(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

for(vals in values_per_thread) {
    for(threads in threads_per_block) {
        data <- read.table(paste0("results/threads/results-vals", vals, "-threads", threads, ".csv"), header=F,  sep=",", stringsAsFactors=F, na.strings="-1")
        names(data) <- c("time")
        data$vals <- vals
        data$threads <- threads
        if (vals == values_per_thread[1] && threads == threads_per_block[[1]]) {
            total_data <- data
        } else {
            total_data <- rbind(total_data, data)
        }
    }
}
total_data %>% group_by(vals, threads) %>% summarise(time=median(time)) %>% as.data.frame() -> total_data

pdf("plots/threads.pdf", width=7, height=7)
ggplot(data = total_data, aes(x = threads, y = vals)) +
  geom_tile(aes(fill = time)) + xlab("Threads Per Block (#)") + ylab("Values Per Thread (#)") +
   scale_x_continuous(trans='log2', breaks=threads_per_block) + scale_y_continuous(trans='log2', breaks=values_per_thread) +
    scale_alpha( trans = "log" )
dev.off()

# different options
files <- list.files('results/options')
for(file in files) {
    data <- read.table(paste0("results/options/", file), header=F,  sep=",", stringsAsFactors=F, na.strings="-1")
    names(data) <- c("time")
    data$pinned <- if(!grepl("--no-pinned-memory", file)) "\\checkmark" else "$\\times$"
    data$coalesced <- if(grepl("--use-coalescing", file)) "\\checkmark" else "$\\times$"
    data$htplacement <- if(grepl("--use-global-ht", file)) "Global" else "Local"
    data$datatypes <- if(grepl("--use-small-datatypes", file)) "Small" else "Full"

    if (file == files[[1]]) {
        total_data <- data
    } else {
        total_data <- rbind(total_data, data)
    }
}
total_data %>% group_by(pinned, coalesced, htplacement, datatypes) %>% summarise(time=median(time)) %>% as.data.frame() %>% arrange(-time) -> total_data



print(xtable::xtable(total_data), include.rownames=FALSE, sanitize.colnames.function = identity, sanitize.text.function = identity)


