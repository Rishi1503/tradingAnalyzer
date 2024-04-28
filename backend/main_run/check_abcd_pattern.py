import numpy as np
import yfinance as yf

def check_abcd_pattern(prices):
    pattern_found = False

    # Minimum number of data points required for the pattern
    min_points = 4

    if len(prices) < min_points:
        return pattern_found

    # Calculate price differences
    price_diff = np.diff(prices)

    for i in range(2, len(prices)):
        if pattern_found:
            break

        for j in range(1, i):
            if pattern_found:
                break

            for k in range(j):
                # Check if the current prices form an ABCD pattern
                if (
                    price_diff[i] > 0  # D to A is an uptrend
                    and price_diff[j] < 0  # A to B is a downtrend
                    and price_diff[k] > 0  # B to C is an uptrend
                    and all(price_diff[x] < 0 for x in range(k + 1, j))  # C to D is a downtrend
                ):
                    pattern_found = True

    return pattern_found

print(check_abcd_pattern(yf.download('CHT')))