#########################################
# Code to remove miscellaneous clusters #
#########################################

import os
import csv

def clean_clusters(input_file, output_file, threshold):
    with open(input_file, 'r', newline='', encoding='utf8') as infile, \
         open(output_file, 'w', newline='', encoding='utf8') as outfile:

        # The part with the csv readers/writers has been adapted from chatGPT
        reader = csv.DictReader(infile, delimiter=',')
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()

        #Filter the lines based on the threshold
        for row in reader:
            try:
                #convert the percentages in floats to avoid errors
                percentage = float(row["Exact Match Percentage"].replace('%', '').strip())
                if percentage > threshold:#filter
                    writer.writerow(row)
            except ValueError:
                continue

def main():
    input_file = "/home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver/data/clusters/analysis/clusters_with_exact_matches.csv"
    output_dir = "/home/noama1/recomendation_system/LLM-Playlist-Recommender-Improver/data/clusters/clean/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "clusters_with_exact_matches.csv")

    clean_clusters(input_file, output_file, threshold=2)
    print(f"Cleaned clusters saved to: {output_file}")

if __name__ == "__main__":
    main()

