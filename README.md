## Narritive Cloze Event Chain Learning with Bert and Enhanced Features


##Train
python bertstory.py

####Arguments
--data: Path of the pickle data file

--output_dir: Output directory for the model files

--device: GPU Id

--sentence: Flag to consider event sentences

--no_context: Flag to ignore event chain and classify only on the basis of candidates

--neeg_dataset: Flag to consider NEEG data format

--story_cloze: Flag to consider Story Cloze data format

--conceptnet: Flag to consider concept net embeddings in input

--candidates: Number of candidates, 5 for MCNC and 2 for Story Cloze Task

--input_size: Number of entries to consider from input dataset

####Examples
MCNC Sentence Dataset

python bertstory.py --data "dataset/gw_extractions_no_rep_no_fin.pickle" --sentence --output_dir output_sentence --device 2 &>> sentence_logs &

NEEG Dataset

python bertstory.py --data "dataset/neeg.pickle" --neeg_dataset --output_dir output_neeg --device 2 &>neeg_logs&

STORY Cloze Task

python bertstory.py --candidates 2 --data "dataset/story_data.pkl" --output_dir output_story --story_cloze --sentence --device 0 --input_size 1800 &>>story_logs&


##Predict

python predict.py

####Arguments

--data: Path of the pickle data file

--model_dir: Directory containing model files

--device: GPU Id

--candidates: Number of candidates, 5 for MCNC and 2 for Story Cloze Task

--sentence: Flag to consider event sentences

--no_context: Flag to ignore event chain and classify only on the basis of candidates

--neeg_dataset: Flag to consider NEEG data format

--story_cloze: Flag to consider Story Cloze data format

--conceptnet: Flag to consider concept net embeddings in input


####Examples

MCNC Sentence Dataset

NEEG Dataset

python predict.py --data "dataset/neeg.pickle" --neeg_dataset --model_dir output_neeg/ --device 2

Story Cloze Dataset

python predict.py --data "dataset/story_winter_val.pkl" --sentence --candidates 2 --model_dir output_story/ --story_cloze --device 2

