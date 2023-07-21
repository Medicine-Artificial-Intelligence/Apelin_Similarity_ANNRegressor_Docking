
import os
import glob
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Enrichment_Factor:
    def __init__(self,data, similarity_measure, pBio_cutoff, active_col, ranked_dataset_percentage_cutoff):
        self.data = data
        self.similarity_measure = similarity_measure
        self.pBio_cutoff = pBio_cutoff
        self.active_col= active_col
        self.ranked_dataset_percentage_cutoff = ranked_dataset_percentage_cutoff
    
    # 1. Enrichment_data
    def get_enrichment_data(self, similarity_measure):
        """
        Calculates x and y values for enrichment plot:
            x - % ranked dataset
            y - % true actives identified

        Parameters
        ----------
        molecules : pandas.DataFrame
            Molecules with similarity values to a query molecule.
        similarity_measure : str
            Column name which will be used to sort the DataFrame．
        pic50_cutoff : float
            pIC50 cutoff value used to discriminate active and inactive molecules.

        Returns
        -------
        pandas.DataFrame
            Enrichment data: Percentage of ranked dataset by similarity vs. percentage of identified true actives.
        """

        # Get number of molecules in data set
        molecules_all = len(self.data)

        # Get number of active molecules in data set
        actives_all = sum(self.data[self.active_col] >= self.pBio_cutoff)

        # Initialize a list that will hold the counter for actives and molecules while iterating through our dataset
        actives_counter_list = []

        # Initialize counter for actives
        actives_counter = 0

        # Note: Data must be ranked for enrichment plots:
        # Sort molecules by selected similarity measure
        self.data.sort_values([similarity_measure], ascending=False, inplace=True)

        # Iterate over the ranked dataset and check each molecule if active (by checking active_col)
        for value in self.data[self.active_col]:
            if value >= self.pBio_cutoff:
                actives_counter += 1
            actives_counter_list.append(actives_counter)

        # Transform number of molecules into % ranked dataset
        molecules_percentage_list = [i / molecules_all for i in range(1, molecules_all + 1)]

        # Transform number of actives into % true actives identified
        actives_percentage_list = [i / actives_all for i in actives_counter_list]

        # Generate DataFrame with x and y values as well as label
        self.enrichment = pd.DataFrame(
            {
                "% ranked dataset": molecules_percentage_list,
                "% true actives identified": actives_percentage_list,
            }
        )

        return self.enrichment
    # 2. Get EF data
    def EF(self):
        self.enrichment_data = {
            similarity_measure: self.get_enrichment_data(similarity_measure)
            for similarity_measure in self.similarity_measure}
        for i in self.similarity_measure:
            print(i)
            display(self.enrichment_data[f"{i}"].head())
            
      
    # 3. Calculate enrichment_factor for dataset
    def calculate_enrichment_factor(self, enrichment):
        """
        Get the experimental enrichment factor for a given percentage of the ranked dataset.

        Parameters
        ----------
        enrichment : pd.DataFrame
            Enrichment data: Percentage of ranked dataset by similarity vs. percentage of
            identified true actives.
        ranked_dataset_percentage_cutoff : float or int
            Percentage of ranked dataset to be included in enrichment factor calculation.

        Returns
        -------
        float
            Experimental enrichment factor.
        """

        # Keep only molecules that meet the cutoff
        enrichment = enrichment[
            enrichment["% ranked dataset"] <= self.ranked_dataset_percentage_cutoff / 100
        ]
        # Get highest percentage of actives and the corresponding percentage of actives
        highest_enrichment = enrichment.iloc[-1]
        enrichment_factor = round(100 * float(highest_enrichment["% true actives identified"]), 1)
        return enrichment_factor
    
    # 4. Calculate random enrichment_factor for dataset
    def calculate_enrichment_factor_random(self):
        """
        Get the random enrichment factor for a given percentage of the ranked dataset.

        Parameters
        ----------
        ranked_dataset_percentage_cutoff : float or int
            Percentage of ranked dataset to be included in enrichment factor calculation.

        Returns
        -------
        float
            Random enrichment factor.
        """

        enrichment_factor_random = round(float(self.ranked_dataset_percentage_cutoff), 1)
        return enrichment_factor_random
    
    # 5. Calculate optimal enrichment_factor for dataset
    def calculate_enrichment_factor_optimal(self):
        """
        Get the optimal random enrichment factor for a given percentage of the ranked dataset.

        Parameters
        ----------
        molecules : pandas.DataFrame
            the DataFrame with all the molecules and pIC50.
        ranked_dataset_percentage_cutoff : float or int
            Percentage of ranked dataset to be included in enrichment factor calculation.
        activity_cutoff: float
            pIC50 cutoff value used to discriminate active and inactive molecules

        Returns
        -------
        float
            Optimal enrichment factor.
        """

        ratio = sum(self.data[self.active_col] >= self.pBio_cutoff) / len(self.data) * 100
        if self.ranked_dataset_percentage_cutoff <= ratio:
            enrichment_factor_optimal = round(100 / ratio * self.ranked_dataset_percentage_cutoff, 1)
        else:
            enrichment_factor_optimal = 100.0
        return enrichment_factor_optimal
    
    # 6. Calculate all enrichment_factor for dataset
    def EF_calculate(self):
        for similarity_measure, enrichment in self.enrichment_data.items():
            enrichment_factor = self.calculate_enrichment_factor(enrichment)
            print(
                f"Experimental EF for {self.ranked_dataset_percentage_cutoff}% of ranked dataset ({similarity_measure}): {enrichment_factor}%"
            )
        enrichment_factor_random = self.calculate_enrichment_factor_random()
        print(
            f"Random EF for {self.ranked_dataset_percentage_cutoff}% of ranked dataset: {enrichment_factor_random}%"
        )
        enrichment_factor_optimal = self.calculate_enrichment_factor_optimal()
        print(
            f"Optimal EF for {self.ranked_dataset_percentage_cutoff}% of ranked dataset: {enrichment_factor_optimal}%"
        )

    # 7. Plot the enrichment data next to the optimal and random enrichment curve!
    def plot_EF(self):
        
        fig, ax = plt.subplots(figsize=(14, 10))

        fontsize = 20

        # Plot enrichment data
        for similarity_measure, enrichment in self.enrichment_data.items():
            ax = enrichment.plot(
                ax=ax,
                x="% ranked dataset",
                y="% true actives identified",
                label=similarity_measure,
                alpha=1,
                linewidth=2,
            )
        ax.set_ylabel("% True actives identified", size=fontsize)
        ax.set_xlabel("% Ranked dataset", size=fontsize)

        # Plot optimal curve: Ratio of actives in datasets
        ratio_actives = sum(self.data[self.active_col] >= self.pBio_cutoff) / len(self.data)
        ax.plot(
            [0, ratio_actives, 1],
            [0, 1, 1],
            label="Optimal curve",
            color="black",
            linestyle="--",
        )

        # Plot random curve
        ax.plot([0, 1], [0, 1], label="Random curve", color="red", linestyle="--")

        plt.tick_params(labelsize=12)
        plt.legend(
            labels= self.similarity_measure+["Optimal", "Random"],
            loc='lower right',
            fontsize= 12,
            labelspacing=0.2,
        )

        # Save plot -- use bbox_inches to include text boxes
        plt.savefig(
            "enrichment_plot.png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )

        plt.show()
