from datetime import date

from nfl4th.data.season import current_target_season


def test_early_in_a_new_season_returns_that_seasons_year():
    assert current_target_season(date(2026, 9, 15)) == 2026


def test_september_first_is_already_the_new_season():
    assert current_target_season(date(2026, 9, 1)) == 2026


def test_end_of_calendar_year_is_still_that_seasons_year():
    assert current_target_season(date(2026, 12, 31)) == 2026


def test_playoffs_in_january_are_still_the_prior_septembers_season():
    assert current_target_season(date(2027, 1, 15)) == 2026


def test_day_before_kickoff_is_still_the_prior_season():
    assert current_target_season(date(2027, 8, 31)) == 2026


def test_offseason_in_june_is_still_the_prior_season():
    assert current_target_season(date(2027, 6, 1)) == 2026
