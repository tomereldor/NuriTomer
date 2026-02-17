"""
Batch market scanner using yfinance download (different endpoint)
This uses yf.download() which may bypass individual ticker rate limits
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Valid biotech tickers (cleaned list - removed obvious invalid ones)
BIOTECH_TICKERS = [
    'ABBV', 'ABSI', 'ABUS', 'ACAD', 'ACLX', 'ADMA', 'AGIO', 'AKBA', 'ALKS',
    'ALNY', 'ALT', 'AMGN', 'ANAB', 'ANNX', 'APGE', 'APLS', 'ARDX', 'ARQT',
    'ARWR', 'AVXL', 'BBIO', 'BCRX', 'BEAM', 'BHVN', 'BIIB', 'BMRN', 'CDNA',
    'CELC', 'CGEM', 'CLDX', 'CMPX', 'COGT', 'CPRX', 'CRSP', 'CRVS', 'CTMX',
    'CYTK', 'DAWN', 'DNLI', 'DVAX', 'DYN', 'EBS', 'ERAS', 'EXAS', 'EXEL',
    'FDMT', 'FOLD', 'GERN', 'GILD', 'GLUE', 'GOSS', 'GRAL', 'HALO', 'IBRX',
    'IDYA', 'IMNM', 'IMRX', 'IMVT', 'INBX', 'INCY', 'INSM', 'IONS', 'IOVA',
    'IRON', 'IRWD', 'IVVD', 'JANX', 'KALV', 'KOD', 'KROS', 'KRYS', 'KURA',
    'KYMR', 'LXEO', 'MDGL', 'MDXG', 'MIRM', 'MLYS', 'MNKD', 'MRNA', 'MYGN',
    'NBIX', 'NRIX', 'NTLA', 'NTRA', 'NUVL', 'NVAX', 'OLMA', 'ORIC', 'PCVX',
    'PGEN', 'PRAX', 'PRME', 'PRTA', 'PTCT', 'PTGX', 'PVLA', 'QURE', 'RAPT',
    'RARE', 'RCUS', 'REGN', 'REPL', 'RGNX', 'RIGL', 'RLAY', 'RNA', 'ROIV',
    'RVMD', 'RXRX', 'RYTM', 'RZLT', 'SANA', 'SION', 'SLNO', 'SMMT', 'SNDX',
    'SPRY', 'SRPT', 'SRRK', 'STOK', 'SVRA', 'SYRE', 'TGTX', 'TNGX', 'TSHA',
    'TVTX', 'TWST', 'TYRA', 'UTHR', 'VCEL', 'VCYT', 'VERA', 'VIR', 'VKTX',
    'VRDN', 'VRTX', 'VSTM', 'XNCR'
]


def batch_scan(
    tickers: list,
    start_date: str = '2024-01-01',
    min_move_pct: float = 0.30,
    output_file: str = 'raw_moves_30pct.csv',
    batch_size: int = 20
) -> pd.DataFrame:
    """
    Scan tickers in batches using yf.download().
    """
    print("=" * 60)
    print(f"BATCH SCANNING {len(tickers)} TICKERS FOR >{min_move_pct*100:.0f}% MOVES")
    print(f"Date range: {start_date} to present")
    print(f"Batch size: {batch_size}")
    print("=" * 60)

    all_events = []

    # Process in batches
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_str = ' '.join(batch)
        print(f"\nBatch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}: {len(batch)} tickers")

        try:
            # Download batch data
            data = yf.download(
                batch_str,
                start=start_date,
                progress=False,
                threads=True,
                timeout=60
            )

            if data.empty:
                print("  No data returned")
                time.sleep(5)
                continue

            # Handle single ticker case (no multi-level columns)
            if len(batch) == 1:
                ticker = batch[0]
                if 'Close' in data.columns:
                    close_df = data[['Close']].copy()
                    close_df.columns = [ticker]
                    data = pd.DataFrame({'Close': close_df})

            # Get close prices
            if 'Close' in data.columns:
                close_data = data['Close']
            elif isinstance(data.columns, pd.MultiIndex):
                close_data = data.xs('Close', level=0, axis=1)
            else:
                print("  Unexpected data format")
                continue

            # Calculate daily returns for each ticker
            for ticker in batch:
                if ticker not in close_data.columns:
                    continue

                prices = close_data[ticker].dropna()
                if len(prices) < 2:
                    continue

                returns = prices.pct_change()
                big_moves = returns[abs(returns) > min_move_pct]

                for date, ret in big_moves.items():
                    if pd.isna(ret):
                        continue
                    move_type = "Gainer" if ret > 0 else "Loser"
                    all_events.append({
                        'Ticker': ticker,
                        'Date': date.strftime('%Y-%m-%d'),
                        'Type': move_type,
                        'Move_%': round(ret * 100, 2),
                        'Price_Event': round(prices.loc[date], 2) if date in prices.index else 0
                    })

            print(f"  Processed, total events so far: {len(all_events)}")

        except Exception as e:
            print(f"  Error: {str(e)[:80]}")

        # Delay between batches
        time.sleep(3)

    # Create final DataFrame
    df = pd.DataFrame(all_events)

    if not df.empty:
        df['Abs_Move'] = df['Move_%'].abs()
        df = df.sort_values('Abs_Move', ascending=False)
        df = df.drop(columns=['Abs_Move'])
        df.to_csv(output_file, index=False)

        print("\n" + "=" * 60)
        print("SCAN COMPLETE")
        print("=" * 60)
        print(f"Total events found: {len(df)}")
        print(f"Unique tickers with events: {df['Ticker'].nunique()}")
        print(f"Gainers: {(df['Type'] == 'Gainer').sum()}")
        print(f"Losers: {(df['Type'] == 'Loser').sum()}")
        if (df['Type'] == 'Gainer').sum() > 0:
            print(f"Biggest gainer: {df[df['Type']=='Gainer'].iloc[0]['Ticker']} +{df[df['Type']=='Gainer']['Move_%'].max():.1f}%")
        if (df['Type'] == 'Loser').sum() > 0:
            print(f"Biggest loser: {df[df['Type']=='Loser'].iloc[0]['Ticker']} {df[df['Type']=='Loser']['Move_%'].min():.1f}%")
        print(f"\nSaved to: {output_file}")
    else:
        print("\nNo events found!")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch scan biotech stocks for large moves")
    parser.add_argument('--min-move', type=float, default=30.0, help='Minimum move percentage')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date')
    parser.add_argument('--output', type=str, default='raw_moves_30pct.csv', help='Output file')
    parser.add_argument('--batch-size', type=int, default=20, help='Tickers per batch')

    args = parser.parse_args()

    batch_scan(
        BIOTECH_TICKERS,
        start_date=args.start_date,
        min_move_pct=args.min_move / 100.0,
        output_file=args.output,
        batch_size=args.batch_size
    )
