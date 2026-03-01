"""Financial data fetcher with detailed error logging."""
import yfinance as yf
from dataclasses import dataclass


@dataclass
class FinancialData:
    market_cap_m: float = None
    current_price: float = None
    cash_position_m: float = None
    cash_runway_months: int = None
    short_percent: float = None
    institutional_ownership: float = None
    analyst_target: float = None
    analyst_rating: str = ""
    error: str = ""
    missing_fields: str = ""


class FinancialDataFetcher:
    def fetch(self, ticker: str) -> FinancialData:
        result = FinancialData()
        missing = []
        try:
            info = yf.Ticker(ticker).info
            if not info or (info.get('regularMarketPrice') is None and info.get('currentPrice') is None):
                result.error = f"No data for {ticker}"
                return result

            if info.get('marketCap'):
                result.market_cap_m = round(info['marketCap'] / 1e6, 1)
            else:
                missing.append('mktcap')

            result.current_price = info.get('currentPrice') or info.get('regularMarketPrice')

            if info.get('totalCash'):
                result.cash_position_m = round(info['totalCash'] / 1e6, 1)
                op_cf = info.get('operatingCashflow')
                if op_cf and op_cf < 0:
                    result.cash_runway_months = int(result.cash_position_m / (abs(op_cf) / 12 / 1e6))
                elif not op_cf:
                    missing.append('opcf')
            else:
                missing.append('cash')

            if info.get('shortPercentOfFloat'):
                result.short_percent = round(info['shortPercentOfFloat'] * 100, 2)
            if info.get('heldPercentInstitutions'):
                result.institutional_ownership = round(info['heldPercentInstitutions'] * 100, 1)
            result.analyst_target = info.get('targetMeanPrice')
            result.analyst_rating = info.get('recommendationKey', '')
            result.missing_fields = ','.join(missing)
        except Exception as e:
            result.error = str(e)[:100]
        return result
