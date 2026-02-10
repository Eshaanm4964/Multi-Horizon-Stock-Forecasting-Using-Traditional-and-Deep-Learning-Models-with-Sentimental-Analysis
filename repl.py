import yfinance as yf

df = yf.download(
    "^GSPC",
    start="2010-01-01",
    end="2024-12-31",
    auto_adjust=True
)

print(df.head())
print(df.tail())
