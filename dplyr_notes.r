library(dplyr)
library(hflights)

data(hflights)
# tbl_df makes a data frame for pretty printing
flights <- tbl_df(hflights)

# print is used with a tbl_df data frame
print(flights, n=20)

## filter ##
# in base R
flights[flights$Month==1 & flights$DayofMonth==1, ]
# dplyr fliter (filters data frame by ROWS)
filter(flights, Month==1, DayofMonth==1)                    # AND
filter(flights, UniqueCarrier=="AA" | UniqueCarrier=="UA")  # OR
filter(flights, UniqueCarrier %in% c("AA", "UA"))           # IN

# filter is much faster than base r
#> system.time(d <- flights[flights$Month==1 & flights$DayofMonth==1, ])
#  user  system elapsed 
#  0.181   0.001   0.183 
#> system.time(d <- filter(flights, Month==1, DayofMonth==1))
#  user  system elapsed 
#  0.023   0.001   0.023 

## select ##
# in base R
flights[, c("DepTime", "ArrTime", "FlightNum")]
# with dplyr select (select COLS from data frame)
select(flights, DepTime, ArrTime, FlightNum)                            # AND
select(flights, Year:DayofMonth, contains("Taxi"), contains("Delay"))   # Year through DayofMonth and COLS containing keywords "Taxi" and "Delay"

## Chaining ##
# from flights, select two COLS (UniqueCarrier and DepDelay)
# then select from this subset observations where DepDelay is 
# greater than 60
flights %>%
    select(UniqueCarrier, DepDelay) %>%
    filter(DepDelay > 60)

## arrange ##
# base R
flights[order(flights$DepDelay), c("UniqueCarrier", "DepDelay")]
# arrange reorders rows
flights %>%
    select(UniqueCarrier, DepDelay) %>%
    arrange(DepDelay)   # arrange(desc(DepDelay))

## mutate ##
# create new vars based on transformations of existing vars
# in base R
flights$Speed <- flights$Distance / flights$AirTime*60
# dplyr::mutate
flights %>%
    select(Distance, AirTime) %>%
    mutate(Speed = Distance/AirTime*60)

# dplyr::mutate and store new variable in flights
flights <- flights %>% mutate(Speed = Distance/AirTime*60)

## summarise ##
# combined with group_by to apply aggregation functions
# to grouped variables

# in base R
d(with(flights, tapply(ArrDelay, Dest, mean, na.rm=TRUE)))
head(aggregate(ArrDelay ~ Dest, flights, mean))

# dplyr::summarise
flights %>%
    group_by(Dest) %>%
    summarise(avg_delay = mean(ArrDelay, na.rm=TRUE))

flights %>%
    group_by(Dest) %>%
    summarise(delay_sd = sd(ArrDelay, na.rm=TRUE))

flights %>%
    group_by(Dest) %>%
    summarise(delay_sd = sd(ArrDelay, na.rm=TRUE)) %>%
    arrange(delay_sd)

# dplyr::summarise_each to apply aggregation to multiple grouped_columns
flights %>%
    group_by(UniqueCarrier) %>%
    summarise_each(funs(mean), Cancelled, Diverted)

flights %>%
    group_by(UniqueCarrier) %>%
    summarise_each(funs(mean(., na.rm = TRUE)), Cancelled, Diverted, ArrDelay, DepDelay) %>%
    arrange(ArrDelay)

flights %>%
    group_by(UniqueCarrier) %>%
    summarise_each(funs(min(., na.rm=TRUE), mean(., na.rm = TRUE), max(., na.rm=TRUE), n = n()), matches("Delay")) %>%
    arrange(DepDelay_mean, ArrDelay_mean)

flights %>%
    group_by(UniqueCarrier) %>%
    summarise(origins = n_distinct(Origin), destinations = n_distinct(Dest))

# tabulation example
# for each destination, show the number of cancelled and not cancelled flights
# in base R
head(table(flights$Dest, flights$Cancelled))
# dplyr
flights %>%
    group_by(Dest) %>%
    select(Cancelled) %>%
    table() %>%
    head()

# a little more overhead in dplyr
# > system.time(head(table(flights$Dest, flights$Cancelled)))
#   user  system elapsed 
#   0.100   0.001   0.101 
# > system.time(flights %>% group_by(Dest) %>% select(Cancelled) %>% table() %>% head())
#   user  system elapsed 
#   0.115   0.000   0.115 

## window functions ##
# http://cran.r-project.org/web/packages/dplyr/vignettes/window-functions.html
# take n inputs and return n outputs

# min_rank()
# max_rank()
# lead()
# lag()
# top_n()
# dense_rank(), cume_dist(), percent_rank(), ntile()
# cumsum(), cummin(), cummax(), cumall(), cumany(), cummean()

# for each carrier, take the top two DepDelays, sort by DepDelay,
# and group by Carrier
flights %>%
    group_by(UniqueCarrier) %>%
    select(Month, DayofMonth, DepDelay) %>%
    top_n(2) %>%
    arrange(UniqueCarrier, desc(DepDelay))

# Monthly counts (tally)and then add a
# change from previous month (lag)
# n comes from tally
flights %>%
    group_by(Month) %>%
    tally() %>%
    mutate(change = n - lag(n))

## Utility Functions ##

# in base R
hflights[sample(nrow(hflights), 5), ]
# random sample from a data frame
flights %>% sample_n(5)

# sample a percentage
flights %>% sample_frac(0.25, replace=TRUE)

str(flights)
# vs
glimpse(flights)

## databases ##
# http://cran.r-project.org/web/packages/dplyr/vignettes/databases.html

# Connects to database as if it were a dataframe
my_db <- src_sqlite("my_db.sqlite3")    # create database object
flights_tbl <- tbl(my_db, "flights")         # database object, table_name
# execute SQL query on database object directly
tbl(my_db, sql("SELECT * FROM hflights LIMIT 100"))
# or use dplyr operations on the dataframe created from the database object's
# table and explain what a comparable SQL query would be.
flights_tbl %>%
    select(UniqueCarrier, DepDelay) %>%
    arrange(desc(DepDelay)) %>%
    explain()
