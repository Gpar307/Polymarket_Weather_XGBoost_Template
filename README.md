###This projects includes a demostration of a temprature prediction market backtesting bot for the [Polymarket prediction market website ](https://polymarket.com/)###

Stracture:

* Templates file:
  * Contains the scripts required for the backtesting. All files with the template indication on their name are meant to be used in combination with a yaml confiquration file for the desired city
  * The mass_scaper.py and mass_scraper_claude_version.py scripts perform the same function. The weather station is hardcoded and should be adjusted for every new station

* YAML files:
  * Contains the configuration yaml files with the city-specific values
  Are meant to be used in combination with the *template.py files

