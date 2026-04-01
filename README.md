## Load Testing Results (Locust)

Tested against: https://checkup-production.up.railway.app

| Users | Requests | Failures | Median (ms) | 95th % (ms) | 99th % (ms) | RPS |
|-------|----------|----------|-------------|-------------|-------------|-----|
| 10    | 29       | 0        | 610         | 1300        | 1400        | 0.5 |
| 50    | 48       | 0        | 730         | 1000        | 1200        | 3.7 |
| 100   | 1083     | 0        | 350         | 730         | 890         | 12.9|

0 failures across all runs. Model handles concurrent requests reliably.