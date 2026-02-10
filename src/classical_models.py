from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet


def arima_model(train, order=(5,1,0)):
    model = ARIMA(train["Close"], order=order)
    return model.fit()


def sarima_model(train, order=(1,1,1), seasonal=(1,1,1,12)):
    model = SARIMAX(
        train["Close"],
        order=order,
        seasonal_order=seasonal
    )
    return model.fit()


def prophet_model(train):
    df = train[["Date", "Close"]].rename(
        columns={"Date": "ds", "Close": "y"}
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True
    )
    model.add_country_holidays(country_name="US")
    model.fit(df)

    return model
