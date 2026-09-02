Quantum Natural Language Processing (QNLP) for Chess Annotation Style Transfer and Blunder Pattern Extraction

Project Overview:

This project explores Quantum Natural Language Processing (QNLP) to analyze chess annotations and extract meaningful patterns such as blunders, pins, forks, and sacrifices.

The pipeline processes chess games and their annotations, cleans and labels the textual data, converts both the natural-language comments and chess board positions into machine-readable representations, and finally builds a dataset suitable for training the QNLP-based model.

Project Pipeline:
The project is divided into the following modules:
1] corpus_loader.py
Extracts the required information from each chess game. For every annotated move, it collects:
1. The board position at that instant
2. The comment associated with the move
3. The game from which the move and comment originated

2] data_cleaner.py
Cleans the raw chess annotations by removing irrelevant information and preparing the comments for analysis.
This includes:
1. Removing clock times
2. Removing engine evaluations
3. Removing other annotation-related junk
4. Removing duplicate comments
5. Counting the number of words in each comment

3] eda_report.py
Performs Exploratory Data Analysis (EDA) on the cleaned dataset.
1. Number of rows
2. Missing comments
3. Comment length distributions
4. General characteristics of the annotations
5. Potential outliers in comment length

4] motif_labeler.py
Classifies each chess comment according to the chess motif it describes.
The current categories are:
blunder
pin
fork
sacrifice
none
The categorical labels are then converted into numerical representations for use by the model. 
For example: pin - 1

5] diagram_parser.py
Converts each annotation sentence into a DisCoCat diagram, providing the grammatical/semantic representation required for the QNLP pipeline.

6] board_encoder.py
Converts each chess board position from FEN (Forsyth-Edwards Notation) into a numerical vector.
Instead of representing a board position as a single FEN string, the board is encoded as a list of numbers that can be provided as input to the machine-learning model.

7] dataset_builder.py
Combines the processed components into the final model-ready dataset.
Each data point is represented as:
(diagram, board_vector, label)
where:
diagram = DisCoCat representation of the chess annotation
board_vector = numerical representation of the chess board position
label = numerical chess-motif label
The resulting dataset is saved and used as the input for training the QNLP model.

8] pipeline.py
Acts as the main pipeline controller.
It runs all the modules in the required sequence, taking the project from raw chess-game data to the final model-ready dataset.
