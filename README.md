# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nurdiyanto-87/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nurdiyanto-87/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI trophy's Exophase earn rate one week after launch, from the house's recording of the trophy page. Pure-stdlib Python 3.13, no network: an earned-vs-tracked counts decomposition (so the dilution when Exophase refreshes the page is priced in), a saturating ceiling fit per trophy, and a damped recent trend — blended by how much history actually moved, with an honest spread for the 4.5-day horizon. On a real backtest (launch-night history vs the live page 8.5 hours later) it scores a pinball loss of 0.346 against the house baseline's 1.385, with 98% coverage of its 90% band.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=silver-quasar-79&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
