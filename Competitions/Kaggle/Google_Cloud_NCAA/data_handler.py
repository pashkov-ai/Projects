import os

import pandas as pd


class DataHandler:

    def __init__(self, sex: str = 'mens'):
        """
        sex = {mens, womens}
        """
        if sex not in {'mens', 'womens'}:
            raise ValueError("Sex parameter should be 'mens' or 'womens'")

        self.data_path = 'data/google-cloud-ncaa-march-madness-2020-division-1-{}-tournament'
        self.data_path = self.data_path.format(sex)
        self.SEX = 'M' if sex == 'mens' else 'W'

    def read_data(self):
        regseason_compact_results_path = '{0}DataFiles_Stage1/{0}RegularSeasonCompactResults.csv'
        regseason_detailed_results_path = '{0}DataFiles_Stage1/{0}RegularSeasonDetailedResults.csv'
        tourney_compact_results_path = '{0}DataFiles_Stage1/{0}NCAATourneyCompactResults.csv'
        tourney_detailed_results_path = '{0}DataFiles_Stage1/{0}NCAATourneyDetailedResults.csv'
        tourney_seeds_path = '{0}DataFiles_Stage1/{0}NCAATourneySeeds.csv'
        tourney_slots_path = '{0}DataFiles_Stage1/{0}NCAATourneySlots.csv'

        data = {'LETTER': self.SEX, 'Tourney': {}, 'RegularSeason': {}, }
        data['RegularSeason']['CompactResults'] = pd.read_csv(
            os.path.join(self.data_path, regseason_compact_results_path.format(self.SEX)))
        data['RegularSeason']['DetailedResults'] = pd.read_csv(
            os.path.join(self.data_path, regseason_detailed_results_path.format(self.SEX)))
        data['Tourney']['CompactResults'] = pd.read_csv(
            os.path.join(self.data_path, tourney_compact_results_path.format(self.SEX)))
        data['Tourney']['DetailedResults'] = pd.read_csv(
            os.path.join(self.data_path, tourney_detailed_results_path.format(self.SEX)))
        data['Tourney']['Seeds'] = pd.read_csv(os.path.join(self.data_path, tourney_seeds_path.format(self.SEX)))
        data['Tourney']['Slots'] = pd.read_csv(os.path.join(self.data_path, tourney_slots_path.format(self.SEX)))

        return data

    def read_submission(self):
        path = os.path.join(self.data_path, '{}SampleSubmissionStage1_2020.csv'.format(self.SEX))
        df = pd.read_csv(path)
        return df

    def make_submission(self, preds):
        submission_df = self.read_submission()
        submission_df['Pred'] = preds
        submission_df.to_csv('submission.csv', index=False)
