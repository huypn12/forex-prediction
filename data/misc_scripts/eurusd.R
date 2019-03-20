library(quantmod)
library(tcltk)

getFX("EUR/USD")
EURUSD

## log difference
eurusdrt <- diff(log(EURUSD))
eurusdrt

X11()
plot(eurusdrt)
capture <- tk_messageBox(message = "something", detail = "here")

plot(acf(eurusdrt, na.action=na.omit))
capture <- tk_messageBox(message = "something", detail = "here")

## Test for AR
eurusdrt.ar <- ar(eurusdrt, na.action=na.omit)
eurusdrt.ar$order
eurusdrt.ar$ar

## Test for MA
eurusdrt.ma <- arima(eurusdrt, order=c(0,0,2))
eurusdrt.ma
plot(acf(eurusdrt.ma$res[-1]))
capture <- tk_messageBox(message = "something", detail = "here")

## Test for ARMA
final.aic <- Inf
final.order <- c(0,0,0)
for (i in 0:10) for (j in 0:10) for (k in 0:2) {
    current.aic <- AIC(arima(eurusdrt, order=c(i, k, j)))
    if (current.aic < final.aic) {
        final.aic <- current.aic
        final.order <- c(i, k, j)
        final.arma <- arima(eurusdrt, order=final.order)
    }
}

final.aic
final.order
final.arma