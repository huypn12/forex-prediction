eurusd <- read.csv("EURUSD_15m_BID_sample.csv")
summary(eurusd)

open <- eurusd$Open 
close <- eurusd$Close 

## Test for ARMA
arimaEst <- function()
{
    final.aic <- Inf
    final.order <- c(0,0,0)
    for (i in 0:10) {
        current.aic <- AIC(arima(open, order=c(i, k, j)))
        if (current.aic < final.aic) {
            final.aic <- current.aic
            final.order <- c(i, k, j)
            final.arma <- arima(open, order=final.order)
        }
    }

    return(final)
}

## estMdl <- arimaEst()

