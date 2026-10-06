## NBA Game Total Prediction using Machine Learning

## 1. Project Overview

This project develops a machine learning system to predict the total number of points scored in an NBA game.

The system uses historical NBA team statistics together with sportsbook betting totals to estimate the final game total.

The main objective is to investigate whether machine learning can produce predictions comparable to the sportsbook betting line.


## 2. Problem Statement

Predicting the total score of an NBA game is challenging because scoring depends on many factors such as team offensive performance, shooting efficiency, rebounding, assists, turnovers, and recent team form.

Sportsbooks provide an estimated total score for each game. This project investigates whether a machine learning model can learn from historical team statistics and the sportsbook line to predict the actual game total.

---

## 3. Dataset

The project uses historical NBA game and betting data covering games from 2006 to 2018.

The data contains:

- Game information
- Team statistics
- Points scored
- Rebounds
- Assists
- Field-goal percentage
- Three-point percentage
- Free-throw percentage
- Turnovers
- Steals
- Blocks
- Sportsbook total line

After preprocessing and filtering, the final machine learning dataset contains:

- 13,840 games
- 1 row per game
- Regular-season games
- Historical team performance features

---

## 4. Feature Engineering

The model uses historical team performance rather than statistics from the game being predicted.

Rolling averages are calculated using previous games to avoid target leakage.

The main features include:

- Average points
- Average rebounds
- Average assists
- Average field-goal percentage
- Average three-point percentage
- Average free-throw percentage
- Average turnovers
- Average steals
- Average blocks
- Sportsbook total line

Additional engineered features include:

- Difference between home and away team statistics
- Combined team statistics

Examples:

- `pts_diff`
- `reb_diff`
- `ast_diff`
- `fg_pct_diff`
- `fg3_pct_diff`
- `combined_pts_avg`
- `combined_reb_avg`
- `combined_ast_avg`
- `combined_tov_avg`

---

## 5. Machine Learning Models

Several approaches were evaluated:

1. Random Forest
2. Random Forest Edge Model
3. Improved Random Forest
4. Gradient Boosting
5. Recent-Form Random Forest

The Improved Random Forest was selected as the final model based on validation performance among the tested approaches.

---

## 6. Final Model

The final model is a Random Forest Regressor.

Configuration:

- Number of trees: 400
- Maximum depth: 10
- Minimum samples per leaf: 5
- Random state: 42

The model predicts:

`actual_total - sportsbook_total_line`

The predicted error is then added to the sportsbook line:

`Predicted Total = Sportsbook Line + Predicted Error`

---

## 7. Train / Validation / Test Split

A chronological split was used instead of a random split.

### Training

Before January 1, 2016

### Validation

January 1, 2016 to December 31, 2017

### Testing

January 1, 2018 onwards

Final test set:

**686 games**

A chronological split was used to better represent real-world prediction, where future games should not influence predictions for earlier games.

---

## 8. Evaluation Metrics

The models were evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual totals.

Lower MAE is better.

### Root Mean Squared Error (RMSE)

Measures prediction error while giving greater weight to larger errors.

Lower RMSE is better.

### R² Score

Measures how much variance in the target is explained by the model.

Higher R² is better.

---

## 9. Final Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Sportsbook Baseline | 14.5924 | 18.3232 | 0.1453 |
| Improved Random Forest | 14.6151 | 18.3216 | 0.1455 |

The Random Forest achieved performance very close to the sportsbook baseline.

The sportsbook achieved a slightly lower MAE, while the Random Forest achieved slightly lower RMSE and slightly higher R².

Therefore, the current model does not provide a meaningful improvement over the sportsbook baseline.

---

## 10. Feature Importance

The most influential features included:

- Sportsbook total line
- Home team three-point percentage
- Combined scoring average
- Away team field-goal percentage
- Away team three-point percentage
- Block difference
- Home team field-goal percentage
- Combined rebounding average

The sportsbook total line was the most important feature.

---

## 11. Prediction System

The project includes an interactive command-line prediction system.

Run:

```bash
cd data
python3 predict.py